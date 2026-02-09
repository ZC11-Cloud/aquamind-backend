"""知识库相关请求/响应模型。"""
from typing import List
from pydantic import BaseModel


class KnowledgeUploadResponse(BaseModel):
    """上传文档后的响应"""
    source_id: str
    filename: str
    chunks_added: int
    message: str = "文档已写入知识库"


class KnowledgeDocumentItem(BaseModel):
    """文档列表项（单库下以 source_id 唯一标识）"""
    source_id: str
    chunk_count: int


class KnowledgeDocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[KnowledgeDocumentItem]
    total: int


class KnowledgeDeleteResponse(BaseModel):
    """删除文档响应（可选，用于返回删除条数）"""
    source_id: str
    chunks_deleted: int
