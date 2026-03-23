"""
知识库路由：文档上传、列表（分页+简介）、详情、删除。
"""
import asyncio
import logging
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, status, UploadFile
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies import get_current_user, get_session
from src.models.document import KnowledgeDocument
from src.models.user import User
from src.schemas.knowledge import (
    KnowledgeUploadResponse,
    KnowledgeDocumentItem,
    KnowledgeDocumentListResponse,
    KnowledgeDeleteResponse,
    KnowledgeDocumentContentResponse,
    KnowledgeSearchResponse,
    KnowledgeSearchHit,
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
    session: AsyncSession = Depends(get_session),
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
        chunks_added, snippet = await asyncio.to_thread(
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
    doc_meta = KnowledgeDocument(
        source_id=stored_name,
        original_filename=file.filename or stored_name,
        storage_path=stored_name,
        summary=snippet or None,
        tags=[],
    )
    try:
        session.add(doc_meta)
        await session.commit()
    except Exception as e:
        logger.exception("文档元数据写入失败: %s", e)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档元数据保存失败",
        ) from e
    return KnowledgeUploadResponse(
        source_id=stored_name,
        filename=file.filename or stored_name,
        chunks_added=chunks_added,
    )


@router.get("/documents", response_model=KnowledgeDocumentListResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
):
    """获取知识库文档列表（分页），含原始文件名、简介截取、标签、chunk 数。"""
    total_result = await session.execute(select(func.count()).select_from(KnowledgeDocument))
    total = total_result.scalar_one()
    offset = (page - 1) * page_size
    result = await session.execute(
        select(KnowledgeDocument)
        .order_by(KnowledgeDocument.create_time.desc())
        .offset(offset)
        .limit(page_size)
    )
    rows = result.scalars().all()
    kb = _get_knowledge_service()
    try:
        source_list = await asyncio.to_thread(kb.list_document_sources)
    except Exception as e:
        logger.warning("Chroma 列表失败，chunk_count 将为 0: %s", e)
        source_list = []
    chunk_map = {x["source_id"]: x["chunk_count"] for x in source_list}
    documents = [
        KnowledgeDocumentItem(
            source_id=r.source_id,
            original_filename=r.original_filename,
            summary=r.summary,
            tags=r.tags or [],
            create_time=r.create_time,
            chunk_count=chunk_map.get(r.source_id, 0),
        )
        for r in rows
    ]
    return KnowledgeDocumentListResponse(
        documents=documents,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/documents/{source_id}", response_model=KnowledgeDocumentItem)
async def get_document(
    source_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """获取单个文档详情（用于前端简介/详情展示）。"""
    if not source_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="source_id 不能为空",
        )
    result = await session.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.source_id == source_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在",
        )
    kb = _get_knowledge_service()
    try:
        source_list = await asyncio.to_thread(kb.list_document_sources)
    except Exception:
        source_list = []
    chunk_map = {x["source_id"]: x["chunk_count"] for x in source_list}
    return KnowledgeDocumentItem(
        source_id=row.source_id,
        original_filename=row.original_filename,
        summary=row.summary,
        tags=row.tags or [],
        create_time=row.create_time,
        chunk_count=chunk_map.get(row.source_id, 0),
    )


@router.get("/documents/{source_id}/content", response_model=KnowledgeDocumentContentResponse)
async def get_document_content(
    source_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """获取文档完整正文，用于前端文档阅读。"""
    if not source_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="source_id 不能为空",
        )
    result = await session.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.source_id == source_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在",
        )
    storage_path = row.storage_path or row.source_id
    upload_dir = Path(KNOWLEDGE_UPLOAD_DIR)
    file_path = upload_dir / storage_path
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档文件不存在或已删除",
        )
    kb = _get_knowledge_service()
    try:
        content = await asyncio.to_thread(kb.get_document_content, str(file_path))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档文件不存在",
        ) from None
    file_ext = Path(row.original_filename).suffix.lower() or ""
    return KnowledgeDocumentContentResponse(
        source_id=row.source_id,
        original_filename=row.original_filename,
        content=content,
        file_ext=file_ext,
    )


@router.get("/search", response_model=KnowledgeSearchResponse)
async def search_documents(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    top_k: int = Query(10, ge=1, le=50, description="返回结果条数"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """搜索知识库文档片段（语义检索）。"""
    query_text = q.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="搜索关键词不能为空",
        )
    kb = _get_knowledge_service()
    try:
        raw_hits = await asyncio.to_thread(kb.search_documents, query_text, top_k)
    except Exception as e:
        logger.exception("知识库搜索失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="知识库搜索失败",
        ) from e
    source_ids = list(
        {
            str(item.get("source_id") or "").strip()
            for item in raw_hits
            if str(item.get("source_id") or "").strip()
        }
    )
    filename_map: dict[str, str] = {}
    if source_ids:
        result = await session.execute(
            select(KnowledgeDocument.source_id, KnowledgeDocument.original_filename).where(
                KnowledgeDocument.source_id.in_(source_ids)
            )
        )
        filename_map = {sid: name for sid, name in result.all()}
    hits = [
        KnowledgeSearchHit(
            source_id=item.get("source_id") or "",
            original_filename=filename_map.get(
                item.get("source_id") or "",
                item.get("source_id") or "未知文档",
            ),
            content=(item.get("content") or "").strip(),
            score=item.get("score"),
        )
        for item in raw_hits
    ]
    return KnowledgeSearchResponse(
        query=query_text,
        total=len(hits),
        hits=hits,
    )


@router.delete("/documents/{source_id}", response_model=KnowledgeDeleteResponse)
async def delete_document(
    source_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """按 source_id 删除文档：向量库 chunk 与文档表记录一并删除。"""
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
    await session.execute(delete(KnowledgeDocument).where(KnowledgeDocument.source_id == source_id))
    await session.commit()
    return KnowledgeDeleteResponse(
        source_id=source_id,
        chunks_deleted=chunks_deleted,
    )
