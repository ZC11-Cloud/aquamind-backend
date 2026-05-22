"""
YOLOv8 图像检测服务：加载 .pt 权重并对上传图片进行目标检测。
推理在线程池中执行，避免阻塞 FastAPI 事件循环。
支持从文件路径或内存字节（numpy 数组）推理，避免临时文件读取失败。
"""
import asyncio
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 全局模型实例与状态，懒加载
_model = None
_model_lock = threading.RLock()
_loaded_weights_path: Optional[str] = None
_current_weights_path: Optional[str] = None
_weights_updated_at: Optional[datetime] = None


def _configure_ultralytics_runtime() -> None:
    backend_dir = Path(__file__).resolve().parents[2]
    config_dir = backend_dir / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    version_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    local_packages = backend_dir / ".cache" / f"python-packages-{version_tag}"
    if local_packages.exists():
        local_packages_str = str(local_packages)
        sys.path[:] = [path for path in sys.path if path != local_packages_str]
        sys.path.insert(0, local_packages_str)


def _log_runtime_backend() -> None:
    try:
        import torch

        logger.info(
            "YOLO runtime: python=%s, torch=%s (%s), cuda=%s, sys_path0=%s",
            sys.executable,
            getattr(torch, "__version__", "unknown"),
            getattr(torch, "__file__", "unknown"),
            torch.cuda.is_available(),
            sys.path[0] if sys.path else "",
        )
    except Exception as exc:
        logger.warning("YOLO runtime dependency check failed: %s", exc)


def _resolve_weights_path(weights_path: str) -> Path:
    """将权重路径解析为绝对路径。"""
    path = Path(weights_path)
    if not path.is_absolute():
        # 相对于项目根目录（backend 的上级）
        base = Path(__file__).resolve().parents[2]
        path = base / weights_path
    return path.resolve()


def _get_effective_weights_path(weights_path: Optional[str] = None) -> str:
    """获取当前生效的权重路径（绝对路径字符串）。"""
    if weights_path:
        return str(_resolve_weights_path(weights_path))
    global _current_weights_path
    if _current_weights_path:
        return _current_weights_path
    from src.settings import YOLO_WEIGHTS_PATH
    return str(_resolve_weights_path(YOLO_WEIGHTS_PATH))


def _load_model(weights_path: str):
    """按给定路径加载 YOLO 模型。"""
    _configure_ultralytics_runtime()
    _log_runtime_backend()
    from ultralytics import YOLO
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO 权重文件不存在: {path}")
    model = YOLO(str(path), task="detect")
    logger.info("YOLOv8 模型加载完成: %s", path)
    return model


def _get_model(weights_path: Optional[str] = None):
    """获取或加载 YOLO 模型（同步，仅在首次调用时加载）。"""
    global _model, _loaded_weights_path, _current_weights_path, _weights_updated_at
    effective_path = _get_effective_weights_path(weights_path)
    with _model_lock:
        if _model is not None and _loaded_weights_path == effective_path:
            return _model
        _model = _load_model(effective_path)
        _loaded_weights_path = effective_path
        _current_weights_path = effective_path
        _weights_updated_at = datetime.now(timezone.utc)
        return _model


def _run_inference(image_path: str, weights_path: str) -> List[dict]:
    """
    在同步上下文中执行 YOLO 推理，返回检测结果列表。
    每项: {"class_name": str, "class_id": int, "confidence": float, "bbox": [x1, y1, x2, y2]}
    """
    model = _get_model(weights_path)
    results = model(image_path, verbose=False)
    return _results_to_detections(results)


def _results_to_detections(results) -> List[dict]:
    """从 YOLO results 提取 detections 列表。"""
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


def _run_inference_from_bytes(image_bytes: bytes, weights_path: str) -> List[dict]:
    """
    从内存中的图片字节推理，避免写临时文件导致的 Image Read Error。
    使用 cv2.imdecode 解码，支持 JPEG/PNG 等常见格式。
    """
    import cv2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("无法从字节解码为图像，请确认为有效的 JPEG/PNG 等格式")
    model = _get_model(weights_path)
    results = model(im, verbose=False)
    return _results_to_detections(results)


def _run_inference_from_bytes_with_annotated(
    image_bytes: bytes, weights_path: str
) -> Tuple[List[dict], bytes]:
    """
    从内存图片推理并返回标注图 JPEG 字节。
    """
    import cv2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("无法从字节解码为图像，请确认为有效的 JPEG/PNG 等格式")
    model = _get_model(weights_path)
    results = model(im, verbose=False)
    detections = _results_to_detections(results)

    if results:
        plotted = results[0].plot()
    else:
        plotted = im
    ok, encoded = cv2.imencode(".jpg", plotted)
    if not ok:
        raise ValueError("无法生成标注图像")
    return detections, encoded.tobytes()


def set_current_weights_path(weights_path: str) -> dict:
    """
    切换当前生效权重并预加载模型，返回当前模型信息。
    """
    global _model, _loaded_weights_path, _current_weights_path, _weights_updated_at
    effective_path = str(_resolve_weights_path(weights_path))
    with _model_lock:
        new_model = _load_model(effective_path)
        _model = new_model
        _loaded_weights_path = effective_path
        _current_weights_path = effective_path
        _weights_updated_at = datetime.now(timezone.utc)
        return get_current_model_info()


def get_current_model_info() -> dict:
    """返回当前生效模型信息。"""
    effective_path = _get_effective_weights_path(None)
    updated_at = _weights_updated_at.isoformat() if _weights_updated_at else None
    return {
        "weights_path": effective_path,
        "weights_name": Path(effective_path).name,
        "updated_at": updated_at,
    }


class YOLODetectionService:
    """YOLOv8 检测服务：异步接口，内部用线程池跑推理。"""

    def __init__(self, weights_path: str):
        self.weights_path = _get_effective_weights_path(weights_path)

    async def detect_from_bytes(self, image_bytes: bytes) -> List[dict]:
        """
        对图片字节数据进行检测。优先从内存解码（cv2.imdecode），避免临时文件读取失败。
        """
        return await asyncio.to_thread(
            _run_inference_from_bytes, image_bytes, self.weights_path
        )

    async def detect_from_path(self, image_path: str) -> List[dict]:
        """对本地图片路径进行检测。"""
        return await asyncio.to_thread(_run_inference, image_path, self.weights_path)

    async def detect_from_bytes_with_annotated(
        self, image_bytes: bytes
    ) -> Tuple[List[dict], bytes]:
        """返回检测结果与标注图 JPEG 字节。"""
        return await asyncio.to_thread(
            _run_inference_from_bytes_with_annotated,
            image_bytes,
            self.weights_path,
        )

    async def reload_model(self, weights_path: str) -> dict:
        """重载模型并切换当前默认权重。"""
        info = await asyncio.to_thread(set_current_weights_path, weights_path)
        self.weights_path = info["weights_path"]
        return info


def get_yolo_service(weights_path: Optional[str] = None):
    """获取 YOLO 检测服务实例。weights_path 为空时从 settings 读取。"""
    effective = _get_effective_weights_path(weights_path)
    return YOLODetectionService(weights_path=effective)
