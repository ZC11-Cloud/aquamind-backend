"""知识库相关请求/响应模型。"""
from datetime import datetime
from typing import List
from pydantic import BaseModel, ConfigDict


class KnowledgeUploadResponse(BaseModel):
    """上传文档后的响应"""
    source_id: str
    filename: str
    chunks_added: int
    message: str = "文档已写入知识库"


class KnowledgeDocumentItem(BaseModel):
    """文档列表项：用于列表与详情展示"""
    source_id: str
    original_filename: str
    summary: str | None = None
    tags: List[str] = []
    create_time: datetime
    chunk_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class KnowledgeDocumentListResponse(BaseModel):
    """文档列表响应（分页）"""
    documents: List[KnowledgeDocumentItem]
    total: int
    page: int = 1
    page_size: int = 20


class KnowledgeDeleteResponse(BaseModel):
    """删除文档响应"""
    source_id: str
    chunks_deleted: int


class KnowledgeDocumentContentResponse(BaseModel):
    """文档完整正文响应（用于文档阅读）"""
    source_id: str
    original_filename: str
    content: str
    file_ext: str  # 如 .md .txt .pdf，供前端决定是否用 Markdown 渲染
