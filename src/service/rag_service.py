"""
RAG 服务：基于知识库检索 + 拼装 prompt，调用现有 AIService 生成回复。
支持构建 citations 供前端 Sources 组件溯源。
"""
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any

from src.service.ai_service import AIService
from src.service.knowledge_service import KnowledgeService
from src.settings import RAG_TOP_K

logger = logging.getLogger(__name__)

# 无检索结果时的兜底说明
NO_CONTEXT_NOTICE = "（当前知识库中暂无与问题直接相关的参考内容，请结合通用知识回答。）"

# 默认 RAG 系统提示前缀
DEFAULT_RAG_SYSTEM_PREFIX = "你是一个智能助手，帮助用户解答问题。请尽量基于以下参考内容回答；若参考内容未涉及，可简要说明并建议用户补充。"

# 引用格式说明（要求 LLM 使用 <sup>N</sup> 格式）
CITATION_FORMAT_INSTRUCTION = (
    "回答中引用参考内容时，请使用 <sup>N</sup> 格式标注，N 为参考编号（1、2、3...），"
    "不要使用「条目[N]」或「[N]」等格式。"
)

# 用于前端 Sources 的 snippet 截取长度
CITATION_SNIPPET_LENGTH = 150


def build_citations_from_docs(docs: List) -> List[Dict[str, Any]]:
    """
    从检索得到的 Document 列表构建 citations，供前端 Sources 组件使用。
    返回 [{ "key": int, "source_id": str, "filename": str, "snippet": str }, ...]
    """
    if not docs:
        return []
    citations = []
    for i, doc in enumerate(docs, 1):
        content = getattr(doc, "page_content", str(doc)).strip()
        if not content:
            continue
        metadata = getattr(doc, "metadata", None) or {}
        source_id = metadata.get("source", "unknown")
        # filename 默认用 source_id，后续可由 qa_service 从 KnowledgeDocument  enriching
        snippet = content[:CITATION_SNIPPET_LENGTH]
        if len(content) > CITATION_SNIPPET_LENGTH:
            snippet += "…"
        citations.append({
            "key": i,
            "source_id": source_id,
            "filename": source_id,
            "snippet": snippet,
        })
    return citations


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

    def build_context_and_citations(self, docs: List) -> Tuple[str, List[Dict[str, Any]]]:
        """
        构建参考上下文与 citations。
        :return: (context_text, citations)，citations 供前端 Sources 使用。
        """
        context = self._build_context_from_docs(docs)
        citations = build_citations_from_docs(docs)
        return context, citations

    def _build_system_prompt(
        self, context: str, base_prompt: Optional[str] = None, include_citation_instruction: bool = True
    ) -> str:
        """拼接带参考内容的 system_prompt，可选包含引用格式说明。"""
        prefix = (base_prompt or DEFAULT_RAG_SYSTEM_PREFIX).strip()
        if include_citation_instruction:
            prefix = f"{prefix}\n\n{CITATION_FORMAT_INSTRUCTION}"
        return f"{prefix}\n\n## 参考内容\n\n{context}"

    async def generate_response_with_rag(
        self,
        question: str,
        messages: List[Dict[str, str]],
        top_k: Optional[int] = None,
        system_prompt_prefix: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        基于知识库检索的问答：先检索再拼 prompt，调用 AIService 生成回复。

        :param question: 当前用户问题（用于检索）。
        :param messages: 对话历史，格式为 [{"role": "user"|"assistant", "content": "..."}]。
        :param top_k: 检索条数，默认使用配置 RAG_TOP_K。
        :param system_prompt_prefix: 系统提示前缀，默认使用 DEFAULT_RAG_SYSTEM_PREFIX。
        :return: (回复文本, citations 列表)，citations 为空时表示未使用知识库或检索无结果。
        """
        retriever = self.knowledge_service.get_retriever(top_k=top_k)
        try:
            docs = await asyncio.to_thread(retriever.invoke, question)
            logger.info("知识库已调用, 检索问题=%s, 命中条数=%d", question[:50], len(docs))
        except Exception as e:
            logger.exception("知识库检索失败: %s", e)
            docs = []
        context, citations = self.build_context_and_citations(docs)
        system_prompt = self._build_system_prompt(context, system_prompt_prefix)
        content = await self.ai_service.generate_response(
            messages=messages,
            system_prompt=system_prompt,
        )
        return content, citations


def create_rag_service(
    ai_service: AIService,
    knowledge_service: KnowledgeService,
) -> RAGService:
    """创建 RAG 服务实例。"""
    return RAGService(
        ai_service=ai_service,
        knowledge_service=knowledge_service,
    )
