"""Agent tools for knowledge retrieval, image recognition, and KB uploads."""
import base64
import logging
from typing import List

from langchain_core.tools import tool

from src.service.knowledge_upload_service import (
    ingest_file_to_knowledge_base,
    resolve_stored_file,
)
from src.settings import QA_ATTACHMENT_UPLOAD_DIR

logger = logging.getLogger(__name__)

NO_CONTEXT_NOTICE = "（当前知识库中暂无与问题直接相关的参考内容。）"


def _format_docs(docs: List) -> str:
    if not docs:
        return NO_CONTEXT_NOTICE
    parts = []
    for i, doc in enumerate(docs, 1):
        content = getattr(doc, "page_content", str(doc)).strip()
        if content:
            parts.append(f"[{i}]\n{content}")
    return "\n\n".join(parts) if parts else NO_CONTEXT_NOTICE


def _format_detections(detections: List[dict]) -> str:
    if not detections:
        return "未检测到任何目标。"
    lines = []
    for i, d in enumerate(detections, 1):
        name = d.get("class_name", "未知")
        conf = d.get("confidence", 0)
        bbox = d.get("bbox", [])
        line = f"{i}. {name}（置信度: {conf:.2%}）"
        if bbox:
            line += f" 边界框: {bbox}"
        lines.append(line)
    return "检测到以下目标：\n" + "\n".join(lines)


def create_agent_tools(knowledge_service, yolo_service, attachments=None, session=None):
    """Create tools bound to the current chat turn."""
    retriever = knowledge_service.get_retriever()
    attachment_map = {
        str(item.get("file_id")): item
        for item in (attachments or [])
        if isinstance(item, dict) and item.get("file_id")
    }

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the AquaMind knowledge base for aquatic species, farming, ecology, or uploaded document knowledge.

        Args:
            query: The user's question or search keywords.
        """
        try:
            docs = retriever.invoke(query)
            logger.info("Agent knowledge search: query=%s, hits=%d", query[:50], len(docs))
            return _format_docs(docs)
        except Exception as e:
            logger.exception("Knowledge search failed: %s", e)
            return f"知识库检索失败: {e}"

    @tool
    async def recognize_image(image_base64: str) -> str:
        """Recognize aquatic organisms or objects in a user-uploaded image.

        Args:
            image_base64: Base64-encoded image data from the current user message.
        """
        if yolo_service is None:
            return "图像识别服务未配置，无法识别图片。"
        b64_clean = (image_base64 or "").replace("\n", "").replace("\r", "").strip()
        if not b64_clean:
            return "未提供有效的 base64 图片数据，无法识别。"
        pad = (4 - len(b64_clean) % 4) % 4
        if pad:
            b64_clean += "=" * pad
        try:
            image_bytes = base64.b64decode(b64_clean, validate=False)
        except Exception as e:
            return f"图片解码失败: {e}"
        try:
            detections = await yolo_service.detect_from_bytes(image_bytes)
            logger.info("Agent image recognition: detections=%d", len(detections))
            return _format_detections(detections)
        except FileNotFoundError as e:
            return f"图像识别服务未就绪: {e}"
        except Exception as e:
            logger.exception("Image recognition failed: %s", e)
            return f"图像识别失败: {e}"

    @tool
    async def upload_file_to_knowledge_base(file_id: str, tags: list[str] | None = None) -> str:
        """Upload one current chat attachment into the knowledge base.

        Use this only when the user explicitly asks to upload, add, import,
        learn, or save an attached PDF/TXT/MD/DOCX file to the knowledge base.

        Args:
            file_id: The file_id from the current message attachment list.
            tags: Optional short category tags for this document.
        """
        if session is None:
            return "上传失败：当前会话未配置数据库写入能力。"
        requested_id = (file_id or "").strip()
        if not requested_id:
            return "上传失败：缺少 file_id。"
        attachment = attachment_map.get(requested_id)
        if not attachment:
            available = ", ".join(attachment_map.keys()) or "无"
            return f"上传失败：该 file_id 不属于本轮消息附件。可用 file_id: {available}"

        try:
            file_path = resolve_stored_file(QA_ATTACHMENT_UPLOAD_DIR, requested_id)
            if not file_path.is_file():
                return "上传失败：附件文件不存在或已被删除。"
            original_filename = str(attachment.get("original_filename") or requested_id)
            result = await ingest_file_to_knowledge_base(
                session=session,
                knowledge_service=knowledge_service,
                file_path=file_path,
                original_filename=original_filename,
                source_id=requested_id,
                tags=tags or [],
            )
            logger.info(
                "Agent uploaded attachment to KB: source_id=%s, chunks=%d",
                result.source_id,
                result.chunks_added,
            )
            return (
                "上传成功："
                f"文件名={result.filename}，source_id={result.source_id}，"
                f"写入片段数={result.chunks_added}。"
            )
        except ValueError as e:
            return f"上传失败：{e}"
        except FileNotFoundError as e:
            return f"上传失败：{e}"
        except Exception as e:
            logger.exception("Agent KB upload failed: %s", e)
            return f"上传失败：文档解析、向量化或元数据保存失败：{e}"

    return [search_knowledge_base, recognize_image, upload_file_to_knowledge_base]
