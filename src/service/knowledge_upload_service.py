import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import KnowledgeDocument
from src.service.knowledge_service import KnowledgeService, LOADER_MAP
from src.settings import KNOWLEDGE_UPLOAD_DIR

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {ext.lstrip(".").lower() for ext in LOADER_MAP.keys()}
MAX_KNOWLEDGE_FILE_SIZE = 20 * 1024 * 1024
MAX_TAGS_PER_DOCUMENT = 10
MAX_TAG_LENGTH = 20


@dataclass(frozen=True)
class StoredUpload:
    file_id: str
    original_filename: str
    file_ext: str
    size: int
    file_path: Path


@dataclass(frozen=True)
class KnowledgeIngestResult:
    source_id: str
    filename: str
    chunks_added: int
    snippet: str


def safe_filename(original: str) -> str:
    safe = re.sub(r"[^\w\-\.]", "_", original)
    return f"{uuid4().hex}_{safe}"


def normalize_tags(raw_tags: str | list[str] | None) -> list[str]:
    if not raw_tags:
        return []
    parsed: object
    if isinstance(raw_tags, str):
        try:
            parsed = json.loads(raw_tags)
        except json.JSONDecodeError:
            parsed = raw_tags.split(",")
    else:
        parsed = raw_tags
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tags must be a JSON array or comma-separated string",
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        value = str(item).strip()
        if not value:
            continue
        if len(value) > MAX_TAG_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tag length cannot exceed {MAX_TAG_LENGTH} characters",
            )
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(value)
        if len(normalized) >= MAX_TAGS_PER_DOCUMENT:
            break
    return normalized


async def save_upload_file(
    file: UploadFile,
    upload_dir: str | Path,
) -> StoredUpload:
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传有效的文件",
        )
    original_filename = file.filename
    ext = Path(original_filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"仅支持格式: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    contents = await file.read()
    if len(contents) > MAX_KNOWLEDGE_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件大小不能超过 20MB",
        )

    target_dir = Path(upload_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_name = safe_filename(original_filename)
    file_path = target_dir / stored_name
    try:
        file_path.write_bytes(contents)
    except OSError as e:
        logger.exception("写入上传文件失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存文件失败",
        ) from e

    return StoredUpload(
        file_id=stored_name,
        original_filename=original_filename,
        file_ext=f".{ext}",
        size=len(contents),
        file_path=file_path,
    )


def resolve_stored_file(upload_dir: str | Path, file_id: str) -> Path:
    safe_id = Path(file_id or "").name
    if not safe_id or safe_id != file_id:
        raise ValueError("无效的附件标识")
    root = Path(upload_dir).resolve()
    path = (root / safe_id).resolve()
    if root not in path.parents and path != root:
        raise ValueError("无效的附件路径")
    return path


async def ingest_file_to_knowledge_base(
    *,
    session: AsyncSession,
    knowledge_service: KnowledgeService,
    file_path: str | Path,
    original_filename: str,
    source_id: str | None = None,
    tags: list[str] | None = None,
) -> KnowledgeIngestResult:
    source = source_id or Path(file_path).name
    normalized_tags = normalize_tags(tags or [])
    existing = await session.execute(
        select(KnowledgeDocument).where(KnowledgeDocument.source_id == source)
    )
    if existing.scalar_one_or_none() is not None:
        raise ValueError(f"文档已存在于知识库: {source}")
    source_path = Path(file_path)
    knowledge_dir = Path(KNOWLEDGE_UPLOAD_DIR)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    target_path = knowledge_dir / source
    if source_path.resolve() != target_path.resolve():
        shutil.copyfile(source_path, target_path)
    else:
        target_path = source_path
    chunks_added, snippet = await asyncio.to_thread(
        knowledge_service.add_document,
        str(target_path),
        source,
    )
    doc_meta = KnowledgeDocument(
        source_id=source,
        original_filename=original_filename or source,
        storage_path=source,
        summary=snippet or None,
        tags=normalized_tags,
    )
    try:
        session.add(doc_meta)
        await session.commit()
    except Exception as e:
        logger.exception("文档元数据写入失败: %s", e)
        await session.rollback()
        raise
    return KnowledgeIngestResult(
        source_id=source,
        filename=original_filename or source,
        chunks_added=chunks_added,
        snippet=snippet,
    )


async def save_and_ingest_knowledge_upload(
    *,
    file: UploadFile,
    tags: str | None,
    session: AsyncSession,
    knowledge_service: KnowledgeService,
    upload_dir: str | Path = KNOWLEDGE_UPLOAD_DIR,
) -> KnowledgeIngestResult:
    stored = await save_upload_file(file, upload_dir)
    normalized_tags = normalize_tags(tags)
    try:
        return await ingest_file_to_knowledge_base(
            session=session,
            knowledge_service=knowledge_service,
            file_path=stored.file_path,
            original_filename=stored.original_filename,
            source_id=stored.file_id,
            tags=normalized_tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("文档解析或向量化失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档解析或向量化失败",
        ) from e
