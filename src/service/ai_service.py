from typing import List, Dict, Any, AsyncGenerator
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from src.service.dashscope_chat_tongyi import ChatTongyiDashScope
from src.service.llm_content_utils import normalize_message_content

import logging
logger = logging.getLogger(__name__)

# 使用 ChatTongyiDashScope，保证 qwen3.5-plus 等走 multimodal-generation 端点
ChatTongyi = ChatTongyiDashScope


class AIService:
    def __init__(self, api_key: str, model_name: str = "qwen-plus"):
        """
        初始化AI服务

        Args:
            api_key: DashScope API密钥
            model_name: 默认使用的模型名称
        """
        self.api_key = api_key
        self.default_model_name = model_name

    @property
    def chat_model(self) -> ChatTongyi:
        """返回用于 Agent 的 ChatTongyi 实例，支持 bind_tools 与 astream。"""
        return self._get_model()

    def _get_model(
        self,
        model_name: str | None = None,
        model_kwargs: Dict[str, Any] | None = None,
    ) -> ChatTongyi:
        """
        根据传入的模型名获取 ChatTongyi 实例；未传或非法时回退到默认模型。
        """
        name = (model_name or self.default_model_name) or "qwen-plus"
        # 这里直接按需创建实例，模型数量有限，性能影响可接受
        logger.info(f"Getting model: {name}")
        kwargs = model_kwargs or {}
        return ChatTongyi(model=name, api_key=self.api_key, model_kwargs=kwargs)  # type: ignore

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "your are a helpful assistant",
        model_name: str | None = None,
        model_kwargs: Dict[str, Any] | None = None,
    ) -> str:
        """
        生成AI回复
        
        Args:
            messages: 消息历史记录，格式为[{"role": "user", "content": "消息内容"}, ...]
            system_prompt: 系统提示词
            
        Returns:
            AI生成的回复内容
        """
        # 构建消息列表
        langchain_messages = []

        # 添加系统提示词
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        # 转换消息历史记录
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        # 调用AI生成回复（保持同步调用以尽量兼容现有行为）
        model = self._get_model(model_name, model_kwargs=model_kwargs)
        response = model.invoke(langchain_messages)

        return normalize_message_content(response.content)

    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "your are a helpful assistant",
        model_name: str | None = None,
        model_kwargs: Dict[str, Any] | None = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        流式生成AI回复，按 token/chunk 逐步 yield 内容。

        Args:
            messages: 消息历史记录，格式为[{"role": "user", "content": "消息内容"}, ...]
            system_prompt: 系统提示词

        Yields:
            每个生成片段的字符串内容（delta）
        """
        langchain_messages = []

        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        model = self._get_model(model_name, model_kwargs=model_kwargs)
        async for chunk in model.astream(langchain_messages):
            reasoning_text = _extract_reasoning_content(chunk)
            if reasoning_text:
                yield {"type": "reasoning_chunk", "content": reasoning_text}
            text = normalize_message_content(chunk.content)
            if text:
                yield {"type": "chunk", "content": text}

    async def generate_short_title(self, user_question: str) -> str:
        """
        根据用户问题生成一句简短的中文对话标题。

        Args:
            user_question: 用户的首条问题内容

        Returns:
            生成的标题，不超过 15 个字
        """
        system_prompt = (
            "根据用户问题生成一句简短的中文对话标题，不超过 15 个字，不要引号、不要标点结尾。"
        )
        messages = [{"role": "user", "content": user_question}]
        result = await self.generate_response(messages, system_prompt)
        return (result or "").strip()[:255]


# 工厂函数，用于创建AIService实例
def create_ai_service(api_key: str, model_name: str = "qwen-plus") -> AIService:
    return AIService(api_key=api_key, model_name=model_name)


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