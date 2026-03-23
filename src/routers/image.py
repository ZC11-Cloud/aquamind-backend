"""
图像识别路由：上传图片，使用 YOLOv8 进行目标检测，并经 LLM 增强输出中文名称与简短描述。
"""
import logging
import os
import re
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, status, UploadFile

from src.dependencies import get_current_user
from src.models.user import User
from src.schemas.detection import DetectionItem, DetectionResponse
from src.schemas.response import ResponseSchema
from src.service.ai_service import create_ai_service
from src.service.yolo_service import get_yolo_service
from src.settings import DASHSCOPE_API_KEY, YOLO_WEIGHTS_PATH

router = APIRouter(prefix="/image", tags=["image"])
logger = logging.getLogger(__name__)

# 允许的图片格式
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# LLM 增强：纯文本调用，使用走 text-generation 的模型（避免无谓多模态计费）
_ai_service = None
if DASHSCOPE_API_KEY:
    _ai_service = create_ai_service(api_key=DASHSCOPE_API_KEY, model_name="qwen3-max")

# LLM 解析：从回复中提取「中文名称」和「简短描述」
_LLM_NAME_PATTERN = re.compile(r"中文名称[：:\s]+(.+?)(?:\n|$)", re.DOTALL)
_LLM_DESC_PATTERN = re.compile(r"简短描述[：:\s]+(.+?)(?:\n|$)", re.DOTALL)


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
        name = name.split("\n")[0].strip()  # 只取第一行
    if desc:
        desc = desc.split("\n")[0].strip()
    return name, desc


async def _enrich_detections_with_llm(raw_detections: List[dict]) -> List[dict]:
    """对检测结果调用 LLM，为置信度最高的一条填充中文名与简短描述。"""
    if not raw_detections or _ai_service is None:
        return raw_detections
    sorted_list = sorted(raw_detections, key=lambda x: x["confidence"], reverse=True)
    best = sorted_list[0]
    user_content = (
        f"第一个物种英文标签：{best['class_name']}，置信度：{best['confidence']}。"
    )
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
        # 为每条 detection 构造带可选增强字段的 dict
        result = []
        for i, d in enumerate(raw_detections):
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
):
    """
    上传一张图片，使用 YOLOv8 进行目标检测，并经 LLM 增强返回中文物种名与简短描述。
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

    # 阶段一：LLM 增强，为检测结果补充中文名与简短描述
    enriched = await _enrich_detections_with_llm(raw_detections)
    detections = [DetectionItem(**d) for d in enriched]
    return ResponseSchema(
        result="success",
        code=200,
        message="检测完成",
        data=DetectionResponse(detections=detections, count=len(detections)).model_dump(),
    )
