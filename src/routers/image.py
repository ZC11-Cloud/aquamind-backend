"""
图像识别路由：上传图片，使用 YOLOv8 进行目标检测。
"""
import logging
import os

from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile

from src.dependencies import get_current_user
from src.models.user import User
from src.schemas.detection import DetectionItem, DetectionResponse
from src.schemas.response import ResponseSchema
from src.service.yolo_service import get_yolo_service
from src.settings import YOLO_WEIGHTS_PATH

router = APIRouter(prefix="/image", tags=["image"])
logger = logging.getLogger(__name__)

# 允许的图片格式
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/detect", response_model=ResponseSchema)
async def detect_objects(
    file: UploadFile = File(..., description="待检测的图片文件"),
    current_user: User = Depends(get_current_user),
):
    """
    上传一张图片，使用训练好的 YOLOv8 模型进行目标检测。
    返回检测到的目标列表（类别名、置信度、边界框）。
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传有效的图片文件",
        )
    ext = os.path.splitext(file.filename)[1][1:].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"仅支持图片格式: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片大小不能超过 20MB",
        )
    try:
        service = get_yolo_service(YOLO_WEIGHTS_PATH)
        raw_detections = await service.detect_from_bytes(contents)
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

    detections = [DetectionItem(**d) for d in raw_detections]
    return ResponseSchema(
        result="success",
        code=200,
        message="检测完成",
        data=DetectionResponse(detections=detections, count=len(detections)).model_dump(),
    )
