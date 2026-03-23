"""
知识库服务：单库、单 Chroma collection。
负责文档加载、分块、向量化与写入/删除。
"""
import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
try:
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings
except ImportError:
    from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.settings import (
    DASHSCOPE_API_KEY,
    CHROMA_PERSIST_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)

# 单库单 collection 名称
DEFAULT_COLLECTION_NAME = "aquamind_kb"

# 简介截取长度（字符数，用于前端展示）
SUMMARY_SNIPPET_LENGTH = 300

# 支持的文档后缀与 Loader 映射（按需扩展）
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


class KnowledgeService:
    """知识库服务：文档接入与向量存储（单库、单 collection）。"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_api_key: Optional[str] = None,
        chunk_size: int = RAG_CHUNK_SIZE,
        chunk_overlap: int = RAG_CHUNK_OVERLAP,
    ):
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        api_key = embedding_api_key or DASHSCOPE_API_KEY
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=api_key,
        )
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

    def _get_loader(self, file_path: str):
        """根据文件后缀返回对应 Loader 类；不支持则返回 None。"""
        ext = Path(file_path).suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if loader_cls is None:
            return None
        # .txt/.md 统一用 UTF-8 解码，避免 Windows 下默认 gbk 导致 UnicodeDecodeError
        if loader_cls is TextLoader:
            return TextLoader(file_path, encoding="utf-8")
        return loader_cls(file_path)

    def _load_documents(self, file_path: str) -> List[Document]:
        """加载单个文件为 Document 列表。"""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        loader = self._get_loader(str(path))
        if loader is None:
            raise ValueError(
                f"不支持的文件类型: {path.suffix}，支持: {list(LOADER_MAP.keys())}"
            )
        return loader.load()

    def add_document(self, file_path: str, source_id: Optional[str] = None) -> Tuple[int, str]:
        """
        将单个文档解析、分块、向量化并写入当前 Chroma collection。

        :param file_path: 文档绝对路径或相对路径。
        :param source_id: 用于检索与删除的文档标识，默认使用文件 basename。
        :return: (写入的 chunk 数量, 内容截取片段用于简介展示)
        """
        file_path = os.path.abspath(file_path)
        source = source_id or os.path.basename(file_path)
        raw_docs = self._load_documents(file_path)
        if not raw_docs:
            logger.warning("文档未解析出内容: %s", file_path)
            return 0, ""
        for doc in raw_docs:
            doc.metadata["source"] = source
        full_text = "".join(d.page_content for d in raw_docs).replace("\r\n", "\n").strip()
        snippet = full_text[:SUMMARY_SNIPPET_LENGTH] if full_text else ""
        if len(full_text) > SUMMARY_SNIPPET_LENGTH:
            snippet += "…"
        splits = self._text_splitter.split_documents(raw_docs)
        if not splits:
            return 0, snippet
        ids = [f"{source}_{i}" for i in range(len(splits))]
        self._vector_store.add_documents(splits, ids=ids)
        logger.info("知识库写入完成: source=%s, chunks=%d", source, len(splits))
        return len(splits), snippet

    def delete_document(self, source_id: str) -> int:
        """
        按文档 source 标识删除该文档在向量库中的全部 chunk。

        :param source_id: 添加时使用的 source（如文件名或传入的 source_id）。
        :return: 删除的 chunk 数量。
        """
        try:
            coll = self._vector_store._collection
        except AttributeError:
            coll = getattr(self._vector_store, "collection", None)
        if coll is None:
            logger.warning("无法获取 Chroma collection，跳过删除")
            return 0
        try:
            # Chroma 按 metadata 查询得到 ids 再删除
            res = coll.get(where={"source": source_id})
            ids = res.get("ids") or []
        except Exception as e:
            logger.exception("按 source 查询 Chroma 失败: %s", e)
            return 0
        if not ids:
            logger.info("未找到 source=%s 的文档", source_id)
            return 0
        self._vector_store.delete(ids=ids)
        logger.info("知识库删除完成: source=%s, chunks=%d", source_id, len(ids))
        return len(ids)

    def list_document_sources(self, limit: int = 10000) -> List[dict]:
        """
        列出当前 collection 中所有文档的 source_id 及对应 chunk 数（用于文档列表接口）。

        :param limit: 从 Chroma 拉取的最大条数，用于去重前采样。
        :return: [{"source_id": str, "chunk_count": int}, ...]
        """
        try:
            coll = self._vector_store._collection
        except AttributeError:
            coll = getattr(self._vector_store, "collection", None)
        if coll is None:
            return []
        try:
            res = coll.get(limit=limit, include=["metadatas"])
            metadatas = res.get("metadatas") or []
            ids = res.get("ids") or []
        except Exception as e:
            logger.exception("Chroma list 失败: %s", e)
            return []
        # 按 source 聚合计数（id 形如 source_i）
        count_by_source: dict = defaultdict(int)
        for meta in metadatas:
            if isinstance(meta, dict) and "source" in meta:
                count_by_source[meta["source"]] += 1
        return [{"source_id": sid, "chunk_count": c} for sid, c in sorted(count_by_source.items())]

    def get_document_content(self, file_path: str) -> str:
        """
        读取文档完整正文，用于前端文档阅读。

        :param file_path: 文档绝对路径。
        :return: 完整文本内容。
        """
        raw_docs = self._load_documents(file_path)
        if not raw_docs:
            return ""
        return "".join(d.page_content for d in raw_docs).replace("\r\n", "\n").strip()

    def search_documents(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """
        在向量库中按语义检索文档片段。

        :param query: 搜索关键词/问题。
        :param top_k: 返回条数。
        :return: [{source_id, content, score}, ...]
        """
        q = (query or "").strip()
        if not q:
            return []
        k = top_k if top_k is not None else RAG_TOP_K
        results = self._vector_store.similarity_search_with_score(q, k=k)
        hits: List[dict] = []
        for doc, score in results:
            source_id = doc.metadata.get("source") if isinstance(doc.metadata, dict) else ""
            content = (doc.page_content or "").strip()
            hits.append(
                {
                    "source_id": source_id or "",
                    "content": content,
                    "score": float(score) if score is not None else None,
                }
            )
        return hits

    def get_retriever(self, top_k: Optional[int] = None, **kwargs):
        """
        返回当前 collection 的 Retriever，供 RAG 链使用。

        :param top_k: 检索条数，默认从 settings 的 RAG_TOP_K 读取时可在外层传入。
        :param kwargs: 透传给 as_retriever(search_kwargs=...)。
        """
        k = top_k if top_k is not None else RAG_TOP_K
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, **kwargs},
        )


def create_knowledge_service(
    persist_directory: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    **kwargs,
) -> KnowledgeService:
    """创建知识库服务实例（单库、单 collection）。"""
    return KnowledgeService(
        persist_directory=persist_directory,
        collection_name=collection_name,
        **kwargs,
    )
