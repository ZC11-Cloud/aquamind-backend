import asyncio
import json
from typing import List, Dict, Any, AsyncGenerator
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from src.service.dashscope_chat_tongyi import ChatTongyiDashScope
from src.service.llm_content_utils import normalize_message_content
from src.settings import AI_REQUEST_TIMEOUT_SECONDS

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
        self._model_cache: dict[str, ChatTongyi] = {}

    @staticmethod
    def _build_model_cache_key(
        model_name: str,
        model_kwargs: Dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> str:
        kwargs = model_kwargs or {}
        # sort_keys 确保相同参数顺序产生稳定 key，便于模型实例复用
        kwargs_key = json.dumps(kwargs, sort_keys=True, ensure_ascii=True)
        return f"{model_name}::streaming={streaming}::{kwargs_key}"

    @property
    def chat_model(self) -> ChatTongyi:
        """返回用于 Agent 的 ChatTongyi 实例，支持 bind_tools 与 astream。"""
        return self._get_model()

    def _get_model(
        self,
        model_name: str | None = None,
        model_kwargs: Dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> ChatTongyi:
        """
        根据传入的模型名获取 ChatTongyi 实例；未传或非法时回退到默认模型。
        """
        name = (model_name or self.default_model_name) or "qwen-plus"
        kwargs = model_kwargs or {}
        cache_key = self._build_model_cache_key(name, kwargs, streaming=streaming)
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached
        model = ChatTongyi(
            model=name,
            api_key=self.api_key,
            model_kwargs=kwargs,
            streaming=streaming,
        )  # type: ignore
        self._model_cache[cache_key] = model
        logger.info("模型实例已创建并缓存: %s, streaming=%s", name, streaming)
        return model

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

        # invoke 为同步调用，放到线程池避免阻塞事件循环
        model = self._get_model(model_name, model_kwargs=model_kwargs)
        response = await asyncio.wait_for(
            asyncio.to_thread(model.invoke, langchain_messages),
            timeout=AI_REQUEST_TIMEOUT_SECONDS,
        )

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

        stream_model_kwargs = dict(model_kwargs or {})
        stream_model_kwargs.setdefault("incremental_output", True)
        model = self._get_model(
            model_name,
            model_kwargs=stream_model_kwargs,
            streaming=True,
        )
        chunk_count = 0
        async for chunk in model.astream(langchain_messages):
            reasoning_text = _extract_reasoning_content(chunk)
            if reasoning_text:
                yield {"type": "reasoning_chunk", "content": reasoning_text}
            text = normalize_message_content(chunk.content)
            if text:
                chunk_count += 1
                yield {"type": "chunk", "content": text}
        logger.info("普通流式输出完成: chunks=%d", chunk_count)

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
