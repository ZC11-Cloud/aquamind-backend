"""
YOLOv8 图像检测服务：加载 .pt 权重并对上传图片进行目标检测。
推理在线程池中执行，避免阻塞 FastAPI 事件循环。
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# 全局模型实例，懒加载
_model = None


def _get_model(weights_path: str):
    """获取或加载 YOLO 模型（同步，仅在首次调用时加载）。"""
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    path = Path(weights_path)
    if not path.is_absolute():
        # 相对于项目根目录（backend 的上级）
        base = Path(__file__).resolve().parents[2]
        path = base / weights_path
    if not path.exists():
        raise FileNotFoundError(f"YOLO 权重文件不存在: {path}")
    _model = YOLO(str(path))
    logger.info("YOLOv8 模型加载完成: %s", path)
    return _model


def _run_inference(image_path: str, weights_path: str) -> List[dict]:
    """
    在同步上下文中执行 YOLO 推理，返回检测结果列表。
    每项: {"class_name": str, "class_id": int, "confidence": float, "bbox": [x1, y1, x2, y2]}
    """
    model = _get_model(weights_path)
    results = model(image_path, verbose=False)
    detections = []
    if not results:
        return detections
    r = results[0]
    names = r.names or {}
    boxes = r.boxes
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    for i in range(len(cls_ids)):
        class_id = int(cls_ids[i])
        class_name = names.get(class_id, f"class_{class_id}")
        confidence = float(conf[i])
        bbox = [float(x) for x in xyxy[i]]
        detections.append({
            "class_name": class_name,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "bbox": bbox,
        })
    return detections


class YOLODetectionService:
    """YOLOv8 检测服务：异步接口，内部用线程池跑推理。"""

    def __init__(self, weights_path: str):
        self.weights_path = weights_path

    async def detect_from_bytes(self, image_bytes: bytes) -> List[dict]:
        """
        对图片字节数据进行检测。会先写入临时文件再调用 YOLO（因 ultralytics 支持 path/ndarray）。
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        try:
            return await asyncio.to_thread(_run_inference, temp_path, self.weights_path)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def detect_from_path(self, image_path: str) -> List[dict]:
        """对本地图片路径进行检测。"""
        return await asyncio.to_thread(_run_inference, image_path, self.weights_path)


def get_yolo_service(weights_path: Optional[str] = None):
    """获取 YOLO 检测服务实例。weights_path 为空时从 settings 读取。"""
    if weights_path is None:
        from src.settings import YOLO_WEIGHTS_PATH
        weights_path = YOLO_WEIGHTS_PATH
    return YOLODetectionService(weights_path=weights_path)
