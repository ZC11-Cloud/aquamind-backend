"""
图像识别路由：上传图片，使用 YOLOv8 进行目标检测，并经 LLM 增强输出中文名称与简短描述。
"""
from datetime import datetime
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, status, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies import get_current_user, get_session
from src.models.image_detection import ImageDetectionHistory
from src.models.user import User
from src.schemas.detection import (
    CurrentModelResponse,
    DetectionHistoryItem,
    DetectionHistoryListResponse,
    DetectionItem,
    DetectionResponse,
)
from src.schemas.response import ResponseSchema
from src.service.ai_service import create_ai_service
from src.service.yolo_service import get_current_model_info, get_yolo_service
from src.settings import DASHSCOPE_API_KEY, UPLOAD_DIR

router = APIRouter(prefix="/image", tags=["image"])
logger = logging.getLogger(__name__)

# 允许的图片格式
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_IMAGE_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MODEL_ALLOWED_EXTENSIONS = {"pt", "onnx", "engine"}
MAX_MODEL_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# LLM 增强：纯文本调用，使用走 text-generation 的模型（避免无谓多模态计费）
_ai_service = None
if DASHSCOPE_API_KEY:
    _ai_service = create_ai_service(api_key=DASHSCOPE_API_KEY, model_name="qwen3-max")

# LLM 解析：从回复中提取「中文名称」和「简短描述」
_LLM_NAME_PATTERN = re.compile(r"中文名称[：:\s]+(.+?)(?:\n|$)", re.DOTALL)
_LLM_DESC_PATTERN = re.compile(r"简短描述[：:\s]+(.+?)(?:\n|$)", re.DOTALL)


def _safe_filename(original: str) -> str:
    safe = re.subn(r"[^\w\-\.]", "_", original)[0]
    return safe or f"file_{uuid4().hex}"


def _to_upload_url(file_path: Path) -> str:
    upload_root = Path(UPLOAD_DIR).resolve()
    relative = file_path.resolve().relative_to(upload_root)
    return f"/uploads/{relative.as_posix()}"


def _pick_top_detection(detections: List[DetectionItem]) -> Tuple[Optional[str], Optional[float]]:
    if not detections:
        return None, None
    best = max(detections, key=lambda d: d.confidence)
    top_name = (best.species_name_zh or "").strip() or best.class_name
    return top_name, best.confidence


def _build_history_item(row: ImageDetectionHistory) -> DetectionHistoryItem:
    detections = [DetectionItem(**item) for item in (row.detections_json or [])]
    return DetectionHistoryItem(
        id=row.id,
        user_id=row.user_id,
        original_image_url=row.original_image_url,
        annotated_image_url=row.annotated_image_url,
        detections=detections,
        top_species_name=row.top_species_name,
        top_confidence=row.top_confidence,
        create_time=row.create_time,
    )


def _weights_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "weights"


def _parse_llm_enrichment(text: str) -> Tuple[Optional[str], Optional[str]]:
    """从 LLM 回复中解析出中文名称和简短描述。"""
    if not text or not text.strip():
        return None, None
    text = text.strip()
    name_match = _LLM_NAME_PATTERN.search(text)
    desc_match = _LLM_DESC_PATTERN.search(text)
    name = name_match.group(1).strip() if name_match else None
    desc = desc_match.group(1).strip() if desc_match else None
    if name:
        name = name.split("\n")[0].strip()
    if desc:
        desc = desc.split("\n")[0].strip()
    if not desc:
        # 兜底：当模型未严格按“简短描述：”格式输出时，尽量从首行文本提取可展示说明
        lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
        candidate_lines: List[str] = []
        for line in lines:
            normalized = line.replace("：", ":")
            if normalized.startswith("中文名称:") or normalized.startswith("简短描述:"):
                continue
            if line == name:
                continue
            candidate_lines.append(line)
        if candidate_lines:
            desc = candidate_lines[0]
    return name, desc


async def _enrich_detections_with_llm(raw_detections: List[dict]) -> List[dict]:
    """对检测结果调用 LLM，为置信度最高的一条填充中文名与简短描述。"""
    if not raw_detections or _ai_service is None:
        return raw_detections
    sorted_list = sorted(raw_detections, key=lambda x: x["confidence"], reverse=True)
    best = sorted_list[0]
    user_content = f"第一个物种英文标签：{best['class_name']}，置信度：{best['confidence']}。"
    system_prompt = (
        "你只根据用户提供的英文物种标签，用两行回复。"
        "第一行格式为：中文名称：xxx\n第二行格式为：简短描述：xxx。"
        "简短描述为一句话介绍该水生生物。不要编造，若无法识别则写未知。"
    )
    try:
        response = await _ai_service.generate_response(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=system_prompt,
        )
        name_zh, desc = _parse_llm_enrichment(response)
        if not desc and response.strip():
            # 二次兜底：即使解析失败，也尽量保留可读的模型解释文本
            desc = response.strip().split("\n")[0][:200]
        result = []
        for d in raw_detections:
            item = {**d, "species_name_zh": None, "description": None}
            if d is best and (name_zh or desc):
                item["species_name_zh"] = name_zh or d.get("class_name")
                item["description"] = desc or ""
            result.append(item)
        return result
    except Exception as e:
        logger.warning("LLM 增强失败，返回原始检测结果: %s", e)
        return [{**d, "species_name_zh": None, "description": None} for d in raw_detections]


@router.post("/detect", response_model=ResponseSchema)
async def detect_objects(
    file: UploadFile = File(..., description="待检测的图片文件"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    上传一张图片，使用 YOLOv8 进行目标检测，并经 LLM 增强返回中文物种名与简短描述。
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请上传有效的图片文件")
    ext = os.path.splitext(file.filename)[1][1:].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"仅支持图片格式: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    contents = await file.read()
    if len(contents) > MAX_IMAGE_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="图片大小不能超过 20MB")
    try:
        service = get_yolo_service()
        raw_detections, annotated_bytes = await service.detect_from_bytes_with_annotated(contents)
    except FileNotFoundError as e:
        logger.warning("YOLO 权重未找到: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="图像识别服务未配置：请将 YOLOv8 权重文件放到指定路径并设置 YOLO_WEIGHTS_PATH",
        ) from e
    except Exception as e:
        logger.exception("YOLO 推理异常: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="图像识别处理失败",
        ) from e

    enriched = await _enrich_detections_with_llm(raw_detections)
    detections = [DetectionItem(**d) for d in enriched]

    today = datetime.now().strftime("%Y%m%d")
    detect_dir = Path(UPLOAD_DIR) / "detections" / str(current_user.id) / today
    detect_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    original_path = detect_dir / f"{token}_original.{ext}"
    annotated_path = detect_dir / f"{token}_annotated.jpg"
    try:
        original_path.write_bytes(contents)
        annotated_path.write_bytes(annotated_bytes)
    except OSError as e:
        logger.exception("保存识别图片失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存识别结果失败",
        ) from e

    original_image_url = _to_upload_url(original_path)
    annotated_image_url = _to_upload_url(annotated_path)
    top_species_name, top_confidence = _pick_top_detection(detections)
    history = ImageDetectionHistory(
        user_id=current_user.id,
        original_image_url=original_image_url,
        annotated_image_url=annotated_image_url,
        detections_json=[item.model_dump() for item in detections],
        top_species_name=top_species_name,
        top_confidence=top_confidence,
    )
    try:
        session.add(history)
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.exception("识别历史写入失败: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="识别结果写入历史失败",
        ) from e

    return ResponseSchema(
        result="success",
        code=200,
        message="检测完成",
        data=DetectionResponse(
            detections=detections,
            count=len(detections),
            annotated_image_url=annotated_image_url,
            original_image_url=original_image_url,
        ).model_dump(),
    )


@router.get("/history", response_model=ResponseSchema)
async def list_detection_history(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    total_result = await session.execute(
        select(func.count()).select_from(ImageDetectionHistory).where(
            ImageDetectionHistory.user_id == current_user.id
        )
    )
    total = total_result.scalar_one()
    offset = (page - 1) * page_size
    rows_result = await session.execute(
        select(ImageDetectionHistory)
        .where(ImageDetectionHistory.user_id == current_user.id)
        .order_by(ImageDetectionHistory.create_time.desc())
        .offset(offset)
        .limit(page_size)
    )
    rows = rows_result.scalars().all()
    payload = DetectionHistoryListResponse(
        records=[_build_history_item(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )
    return ResponseSchema(
        result="success",
        code=200,
        message="获取识别历史成功",
        data=payload.model_dump(),
    )


@router.get("/history/{history_id}", response_model=ResponseSchema)
async def get_detection_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(ImageDetectionHistory).where(
            ImageDetectionHistory.id == history_id,
            ImageDetectionHistory.user_id == current_user.id,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="历史记录不存在")
    return ResponseSchema(
        result="success",
        code=200,
        message="获取识别历史详情成功",
        data=_build_history_item(row).model_dump(),
    )


@router.delete("/history/{history_id}", response_model=ResponseSchema)
async def delete_detection_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        delete(ImageDetectionHistory).where(
            ImageDetectionHistory.id == history_id,
            ImageDetectionHistory.user_id == current_user.id,
        )
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="历史记录不存在")
    await session.commit()
    return ResponseSchema(
        result="success",
        code=200,
        message="删除识别历史成功",
        data={"id": history_id},
    )


@router.post("/model/upload", response_model=ResponseSchema)
async def upload_model_weights(
    file: UploadFile = File(..., description="YOLO 模型文件(.pt/.onnx/.engine)"),
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请上传有效的权重文件")
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in MODEL_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"仅支持模型格式: {', '.join(sorted(MODEL_ALLOWED_EXTENSIONS))}",
        )
    contents = await file.read()
    if len(contents) > MAX_MODEL_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="模型文件大小不能超过 500MB")

    model_dir = _weights_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}_{_safe_filename(file.filename)}"
    stored_path = model_dir / stored_name
    try:
        stored_path.write_bytes(contents)
    except OSError as e:
        logger.exception("保存权重文件失败: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="保存权重文件失败") from e

    service = get_yolo_service()
    try:
        info = await service.reload_model(str(stored_path))
    except Exception as e:
        logger.exception("加载新模型失败: %s", e)
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("删除失败权重文件失败: %s", stored_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="模型加载失败，请检查权重文件是否可用",
        ) from e

    payload = CurrentModelResponse(**info)
    return ResponseSchema(
        result="success",
        code=200,
        message="模型上传成功并已切换为默认模型",
        data=payload.model_dump(),
    )


@router.get("/model/current", response_model=ResponseSchema)
async def get_current_model(
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    payload = CurrentModelResponse(**get_current_model_info())
    return ResponseSchema(
        result="success",
        code=200,
        message="获取当前模型成功",
        data=payload.model_dump(),
    )
