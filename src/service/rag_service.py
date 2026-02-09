"""
RAG 服务：基于知识库检索 + 拼装 prompt，调用现有 AIService 生成回复。
"""
import asyncio
import logging
from typing import List, Dict, Optional

from src.service.ai_service import AIService
from src.service.knowledge_service import KnowledgeService
from src.settings import RAG_TOP_K

logger = logging.getLogger(__name__)

# 无检索结果时的兜底说明
NO_CONTEXT_NOTICE = "（当前知识库中暂无与问题直接相关的参考内容，请结合通用知识回答。）"

# 默认 RAG 系统提示前缀
DEFAULT_RAG_SYSTEM_PREFIX = "你是一个智能助手，帮助用户解答问题。请尽量基于以下参考内容回答；若参考内容未涉及，可简要说明并建议用户补充。"


class RAGService:
    """RAG 服务：检索知识库 → 拼装 system_prompt → 调用 AIService 生成回复。"""

    def __init__(
        self,
        ai_service: AIService,
        knowledge_service: KnowledgeService,
    ):
        self.ai_service = ai_service
        self.knowledge_service = knowledge_service

    def _build_context_from_docs(self, docs: List) -> str:
        """将检索得到的 Document 列表拼成参考上下文字符串。"""
        if not docs:
            return NO_CONTEXT_NOTICE
        parts = []
        for i, doc in enumerate(docs, 1):
            content = getattr(doc, "page_content", str(doc)).strip()
            if content:
                parts.append(f"[{i}]\n{content}")
        return "\n\n".join(parts) if parts else NO_CONTEXT_NOTICE

    def _build_system_prompt(self, context: str, base_prompt: Optional[str] = None) -> str:
        """拼接带参考内容的 system_prompt。"""
        prefix = (base_prompt or DEFAULT_RAG_SYSTEM_PREFIX).strip()
        return f"{prefix}\n\n## 参考内容\n\n{context}"

    async def generate_response_with_rag(
        self,
        question: str,
        messages: List[Dict[str, str]],
        top_k: Optional[int] = None,
        system_prompt_prefix: Optional[str] = None,
    ) -> str:
        """
        基于知识库检索的问答：先检索再拼 prompt，调用 AIService 生成回复。

        :param question: 当前用户问题（用于检索）。
        :param messages: 对话历史，格式为 [{"role": "user"|"assistant", "content": "..."}]。
        :param top_k: 检索条数，默认使用配置 RAG_TOP_K。
        :param system_prompt_prefix: 系统提示前缀，默认使用 DEFAULT_RAG_SYSTEM_PREFIX。
        :return: 模型生成的回复文本。
        """
        retriever = self.knowledge_service.get_retriever(top_k=top_k)
        # 检索为同步调用，放入线程池避免阻塞事件循环
        try:
            docs = await asyncio.to_thread(retriever.invoke, question)
        except Exception as e:
            logger.exception("知识库检索失败: %s", e)
            docs = []
        context = self._build_context_from_docs(docs)
        system_prompt = self._build_system_prompt(context, system_prompt_prefix)
        return await self.ai_service.generate_response(
            messages=messages,
            system_prompt=system_prompt,
        )


def create_rag_service(
    ai_service: AIService,
    knowledge_service: KnowledgeService,
) -> RAGService:
    """创建 RAG 服务实例。"""
    return RAGService(
        ai_service=ai_service,
        knowledge_service=knowledge_service,
    )
