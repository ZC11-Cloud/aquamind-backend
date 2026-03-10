"""
QA 消息图片存储：将 base64 保存为本地文件并返回可访问 URL。
当前为本地文件存储，后期可替换为 OSS 上传逻辑，接口保持不变（返回 URL）。
"""
import base64
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 根据文件头简单判断扩展名
JPEG_HEADER = b"\xff\xd8\xff"
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def save_qa_image_base64_to_file(
    image_base64: str,
    upload_dir: str,
    url_prefix: str = "/uploads",
) -> Optional[str]:
    """
    将 base64 图片保存到 upload_dir/qa/YYYY/MM/DD/<uuid>.<ext>，
    返回前端可访问的相对 URL，如 /uploads/qa/2026/03/10/xxx.jpg。

    :param image_base64: 纯 base64 字符串（无 data:image/... 前缀）
    :param upload_dir: 上传根目录（与 main 中 StaticFiles 的 directory 一致）
    :param url_prefix: URL 前缀，默认 /uploads
    :return: 相对 URL，失败返回 None
    """
    if not (image_base64 or "").strip():
        return None
    try:
        b64_clean = (image_base64 or "").replace("\n", "").replace("\r", "").strip()
        pad = (4 - len(b64_clean) % 4) % 4
        if pad != 4:
            b64_clean += "=" * pad
        raw = base64.b64decode(b64_clean, validate=False)
    except Exception as e:
        logger.warning("QA image base64 decode failed: %s", e)
        return None
    if not raw:
        return None
    # 根据魔数选择扩展名
    ext = "jpg"
    if raw[:3] == JPEG_HEADER:
        ext = "jpg"
    elif raw[:8] == PNG_HEADER:
        ext = "png"
    # 按日期分目录，便于后期按日期清理或迁移 OSS
    from datetime import datetime
    now = datetime.utcnow()
    subdir = os.path.join("qa", str(now.year), f"{now.month:02d}", f"{now.day:02d}")
    save_dir = Path(upload_dir) / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.{ext}"
    file_path = save_dir / name
    try:
        file_path.write_bytes(raw)
    except Exception as e:
        logger.warning("QA image file write failed: %s", e)
        return None
    # 返回 URL 路径（与 StaticFiles mount 一致）
    relative = f"{url_prefix}/{subdir.replace(os.sep, '/')}/{name}"
    return relative
