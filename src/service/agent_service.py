"""
Agent 编排服务：LangGraph ReAct 风格循环，支持工具调用与流式输出。
将知识库检索、图像识别作为工具，由 LLM 决定是否调用；最终回复以流式 yield。
"""
import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool

from src.service.ai_service import AIService
from src.service.dashscope_chat_tongyi import is_multimodal_model
from src.service.knowledge_service import KnowledgeService
from src.service.llm_content_utils import normalize_message_content
from src.tools.agent_tools import create_agent_tools

logger = logging.getLogger(__name__)

# 默认系统提示（Agent 模式下）
DEFAULT_AGENT_SYSTEM_PROMPT = """你是一个水生生物智能助手（AquaMind）。你可以：
1. 使用 search_knowledge_base：当用户询问物种、养殖、生态等知识时，先检索知识库再回答。
2. 使用 recognize_image：当用户上传了图片并希望识别其中的水生生物时，先调用图像识别再结合结果回答。

请根据用户意图决定是否调用工具。若用户仅做一般对话，可直接回答；若需要专业知识或识图，请先调用相应工具再综合回答。"""


def _extract_reasoning_content(chunk: Any) -> str:
    """尽量从 chunk 的不同字段中提取 reasoning_content。"""
    direct = getattr(chunk, "reasoning_content", None)
    if isinstance(direct, str) and direct:
        return direct

    additional_kwargs = getattr(chunk, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        rc = additional_kwargs.get("reasoning_content")
        if isinstance(rc, str) and rc:
            return rc

    response_metadata = getattr(chunk, "response_metadata", None)
    if isinstance(response_metadata, dict):
        rc = response_metadata.get("reasoning_content")
        if isinstance(rc, str) and rc:
            return rc
        generation_info = response_metadata.get("generation_info")
        if isinstance(generation_info, dict):
            rc = generation_info.get("reasoning_content")
            if isinstance(rc, str) and rc:
                return rc

    return ""


def _dict_to_langchain_messages(messages_history: List[Dict[str, str]]) -> List[BaseMessage]:
    """将 [{"role":"user"|"assistant","content":"..."}] 转为 LangChain 消息列表。"""
    out: List[BaseMessage] = []
    for msg in messages_history:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _get_tool_by_name(tools: List[BaseTool], name: str) -> Optional[BaseTool]:
    for t in tools:
        if t.name == name:
            return t
    return None


async def _invoke_tool(tool: BaseTool, args: dict) -> str:
    """同步或异步调用工具，返回 content 字符串。"""
    if hasattr(tool, "ainvoke") and asyncio.iscoroutinefunction(tool.ainvoke):
        result = await tool.ainvoke(args)
    else:
        result = await asyncio.to_thread(tool.invoke, args)
    if isinstance(result, str):
        return result
    return str(result)


class AgentService:
    """ReAct Agent：绑定工具到 ChatTongyi，循环执行 tool_calls 直至得到最终回复，并支持流式输出。"""

    def __init__(
        self,
        ai_service: AIService,
        knowledge_service: KnowledgeService,
        yolo_service: Optional[Any] = None,
    ):
        self.ai_service = ai_service
        self.knowledge_service = knowledge_service
        self.yolo_service = yolo_service
        self._tools = create_agent_tools(knowledge_service, yolo_service)
        self._tools_by_name = {t.name: t for t in self._tools}

    def _model_with_tools(
        self,
        model_name: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        return self.ai_service._get_model(
            model_name, model_kwargs=model_kwargs, streaming=True
        ).bind_tools(self._tools)

    @staticmethod
    def _as_data_url(image_base64: str) -> str:
        img = (image_base64 or "").strip()
        if img.startswith("data:image/"):
            return img
        return f"data:image/jpeg;base64,{img}"

    async def run_agent_stream(
        self,
        messages_history: List[Dict[str, str]],
        current_user_content: str,
        system_prompt: Optional[str] = None,
        inject_messages: Optional[List[BaseMessage]] = None,
        model_name: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        image_base64: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        运行 Agent 循环，流式产出事件：
        - {"type":"reasoning_chunk","content":"..."}
        - {"type":"chunk","content":"..."}
        messages_history: 历史对话 [{"role","content"}]
        current_user_content: 当前轮用户输入（可含注入的上下文描述）
        system_prompt: 系统提示，默认 DEFAULT_AGENT_SYSTEM_PROMPT
        inject_messages: 在当轮用户消息前插入的额外消息（如注入的 RAG/图像结果）
        """
        prompt = system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT
        langchain_messages: List[BaseMessage] = [
            SystemMessage(content=prompt),
            *_dict_to_langchain_messages(messages_history),
        ]
        if inject_messages:
            langchain_messages.extend(inject_messages)
        if image_base64 and is_multimodal_model(model_name):
            # 多模态模型：在用户文本外附上原图，让模型直接看到图片。
            langchain_messages.append(
                HumanMessage(
                    content=[
                        {"text": current_user_content},
                        {"image": self._as_data_url(image_base64)},
                    ]
                )
            )
        else:
            langchain_messages.append(HumanMessage(content=current_user_content))

        stream_model_kwargs = dict(model_kwargs or {})
        stream_model_kwargs.setdefault("incremental_output", True)
        model = self._model_with_tools(model_name, model_kwargs=stream_model_kwargs)
        response: Optional[BaseMessage] = None
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            accumulated: Optional[AIMessageChunk] = None
            chunk_count = 0

            async for chunk in model.astream(langchain_messages):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                reasoning_text = _extract_reasoning_content(chunk)
                if reasoning_text:
                    yield {"type": "reasoning_chunk", "content": reasoning_text}
                # 真流式：有文本内容就立即 yield
                text = normalize_message_content(chunk.content)
                if text:
                    chunk_count += 1
                    yield {"type": "chunk", "content": text}
                accumulated = chunk if accumulated is None else accumulated + chunk
            logger.info("Agent 第 %d 轮流式输出完成: chunks=%d", iteration, chunk_count)

            response = accumulated
            if response is None:
                break

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            logger.info("Agent 第 %d 轮: %d 个 tool_calls", iteration, len(tool_calls))
            # 合并后的 chunk 可直接 append（AIMessageChunk 是 BaseMessage 子类）
            langchain_messages.append(response)

            for tc in tool_calls:
                if isinstance(tc, dict):
                    raw_name = tc.get("name")
                    raw_args = tc.get("args")
                    raw_id = tc.get("id")
                else:
                    raw_name = getattr(tc, "name", None)
                    raw_args = getattr(tc, "args", None)
                    raw_id = getattr(tc, "id", None)
                name = raw_name if isinstance(raw_name, str) else None
                args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
                tid = raw_id if isinstance(raw_id, str) else ""
                tool = _get_tool_by_name(self._tools, name) if name else None
                if not tool:
                    content = f"未知工具: {name}"
                else:
                    try:
                        content = await _invoke_tool(tool, args)
                    except Exception as e:
                        logger.exception("工具执行失败 %s: %s", name, e)
                        content = f"工具执行失败: {e}"
                langchain_messages.append(ToolMessage(content=content, tool_call_id=tid))

        if response is None:
            yield {"type": "chunk", "content": "抱歉，未能生成回复。"}
            return


def create_agent_service(
    ai_service: AIService,
    knowledge_service: KnowledgeService,
    yolo_service: Optional[Any] = None,
) -> AgentService:
    """创建 Agent 服务实例。yolo_service 为 None 时，recognize_image 工具仍存在但会返回“服务未配置”。"""
    return AgentService(
        ai_service=ai_service,
        knowledge_service=knowledge_service,
        yolo_service=yolo_service,
    )
