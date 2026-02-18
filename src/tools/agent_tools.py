"""
Agent 工具：search_knowledge_base、recognize_image。
封装 knowledge_service 与 yolo_service，供 LangGraph ReAct Agent 调用。
"""
import base64
import logging
from typing import List

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 无检索结果时的兜底说明（与 rag_service 一致）
NO_CONTEXT_NOTICE = "（当前知识库中暂无与问题直接相关的参考内容。）"


def _format_docs(docs: List) -> str:
    """将检索得到的 Document 列表拼成一段参考文本。"""
    if not docs:
        return NO_CONTEXT_NOTICE
    parts = []
    for i, doc in enumerate(docs, 1):
        content = getattr(doc, "page_content", str(doc)).strip()
        if content:
            parts.append(f"[{i}]\n{content}")
    return "\n\n".join(parts) if parts else NO_CONTEXT_NOTICE


def _format_detections(detections: List[dict]) -> str:
    """将 YOLO 检测结果格式化为一段文字描述。"""
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


def create_agent_tools(knowledge_service, yolo_service):
    """
    创建 Agent 使用的两个工具：search_knowledge_base、recognize_image。
    knowledge_service: KnowledgeService 实例
    yolo_service: YOLODetectionService 实例（或 None，则 recognize_image 返回“服务未配置”）
    """
    retriever = knowledge_service.get_retriever()

    @tool
    def search_knowledge_base(query: str) -> str:
        """根据用户问题检索水生生物、物种、养殖等知识库文档。用于回答知识类问题时查询相关参考内容。"""
        try:
            docs = retriever.invoke(query)
            logger.info("Agent 知识库检索: query=%s, 命中=%d", query[:50], len(docs))
            return _format_docs(docs)
        except Exception as e:
            logger.exception("知识库检索失败: %s", e)
            return f"知识库检索失败: {e}"

    @tool
    async def recognize_image(image_base64: str) -> str:
        """当用户上传了水生生物图片、需要识别物种或图中目标时使用。输入为 base64 编码的图片数据。"""
        if yolo_service is None:
            return "图像识别服务未配置，无法识别图片。"
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return f"图片解码失败: {e}"
        try:
            detections = await yolo_service.detect_from_bytes(image_bytes)
            logger.info("Agent 图像识别: 检测到 %d 个目标", len(detections))
            return _format_detections(detections)
        except FileNotFoundError as e:
            return f"图像识别服务未就绪: {e}"
        except Exception as e:
            logger.exception("图像识别失败: %s", e)
            return f"图像识别失败: {e}"

    return [search_knowledge_base, recognize_image]
