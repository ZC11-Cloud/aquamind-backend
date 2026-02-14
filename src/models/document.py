"""知识库文档表：存储文档元数据（基础信息、简介、分类标签），与 Chroma source_id 对应。"""
from datetime import datetime
from typing import List

from sqlalchemy import Integer, DateTime, String, Text, text, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.models import Base


class KnowledgeDocument(Base):
    """
    知识库文档元数据表。
    source_id 与 Chroma 中 metadata["source"] 一致，用于关联向量与展示。
    MySQL 无原生数组类型，分类标签使用 JSON 存储（如 ["水产", "养殖"]）。
    """

    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(512), default=None)
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    tags: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    create_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=func.now(),
    )
