"""
知识库路由：文档上传、列表、删除。
"""
import asyncio
import logging
import os
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile

from src.dependencies import get_current_user
from src.models.user import User
from src.schemas.knowledge import (
    KnowledgeUploadResponse,
    KnowledgeDocumentItem,
    KnowledgeDocumentListResponse,
    KnowledgeDeleteResponse,
)
from src.service.knowledge_service import (
    create_knowledge_service,
    KnowledgeService,
    LOADER_MAP,
)
from src.settings import KNOWLEDGE_UPLOAD_DIR

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {ext.lstrip(".").lower() for ext in LOADER_MAP.keys()}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def _safe_filename(original: str) -> str:
    """生成唯一且安全的存储文件名。"""
    safe = re.subn(r"[^\w\-\.]", "_", original)[0]
    return f"{uuid4().hex}_{safe}"


def _get_knowledge_service() -> KnowledgeService:
    return create_knowledge_service()


@router.post("/upload", response_model=KnowledgeUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="知识库文档（PDF/TXT/MD）"),
    current_user: User = Depends(get_current_user),
):
    """上传文档：保存到知识库目录并触发解析、分块、向量化入库。"""
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传有效的文件",
        )
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"仅支持格式: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件大小不能超过 20MB",
        )
    upload_dir = Path(KNOWLEDGE_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = _safe_filename(file.filename)
    file_path = upload_dir / stored_name
    try:
        file_path.write_bytes(contents)
    except OSError as e:
        logger.exception("写入知识库文件失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存文件失败",
        ) from e
    kb = _get_knowledge_service()
    try:
        chunks_added = await asyncio.to_thread(
            kb.add_document,
            str(file_path),
            source_id=stored_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        logger.exception("知识库写入失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档解析或向量化失败",
        ) from e
    return KnowledgeUploadResponse(
        source_id=stored_name,
        filename=file.filename,
        chunks_added=chunks_added,
    )


@router.get("/documents", response_model=KnowledgeDocumentListResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
):
    """获取知识库文档列表（按 source_id 及 chunk 数）。"""
    kb = _get_knowledge_service()
    try:
        items = await asyncio.to_thread(kb.list_document_sources)
    except Exception as e:
        logger.exception("知识库列表失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文档列表失败",
        ) from e
    docs = [KnowledgeDocumentItem(source_id=x["source_id"], chunk_count=x["chunk_count"]) for x in items]
    return KnowledgeDocumentListResponse(documents=docs, total=len(docs))


@router.delete("/documents/{source_id}", response_model=KnowledgeDeleteResponse)
async def delete_document(
    source_id: str,
    current_user: User = Depends(get_current_user),
):
    """按 source_id 删除该文档在向量库中的全部 chunk。"""
    if not source_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="source_id 不能为空",
        )
    kb = _get_knowledge_service()
    try:
        chunks_deleted = await asyncio.to_thread(kb.delete_document, source_id)
    except Exception as e:
        logger.exception("知识库删除失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除失败",
        ) from e
    return KnowledgeDeleteResponse(
        source_id=source_id,
        chunks_deleted=chunks_deleted,
    )
