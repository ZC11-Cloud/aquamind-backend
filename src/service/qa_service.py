import asyncio
import base64
import json
import logging
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, delete, update
from typing import List, Optional, TYPE_CHECKING, AsyncGenerator, Dict, Any

from src.models import AsyncSessionFactory
from src.models.qa import QaConversation, QaMessage
from src.models.document import KnowledgeDocument
from src.schemas.qa import QaConversationCreate, QaMessageCreate
from src.service.ai_service import AIService
from src.service.rag_service import (
    build_citations_from_docs,
    CITATION_FORMAT_INSTRUCTION,
    filter_citations_by_referenced,
    normalize_citation_format,
)
from src.settings import UPLOAD_DIR
from src.utils.qa_image import save_qa_image_base64_to_file

logger = logging.getLogger(__name__)
DEFAULT_SUGGESTION_LIMIT = 3
MAX_SUGGESTION_LIMIT = 5
FALLBACK_SUGGESTIONS = [
    "你可以用一个更简单的例子再解释一次吗？",
    "如果我要立刻执行，下一步最先做什么？",
    "这个方案有哪些常见误区需要避免？",
    "请给我一个分步骤的检查清单。",
    "还有哪些可选方案，分别适合什么场景？",
]


def _normalize_suggestion_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^\s*(?:[-*•]|\d+[\.、\)]|[（(]?\d+[）)])\s*", "", value)
    value = value.replace("`", "")
    value = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", value)
    value = re.sub(r"[#>*_~]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > 60:
        value = value[:60].rstrip("，。！？!?,;； ") + "？"
    return value


def _extract_suggestions_from_llm(raw_content: str) -> List[str]:
    if not raw_content:
        return []

    candidates: List[str] = []
    content = raw_content.strip()

    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None

    if isinstance(parsed, list):
        candidates = [str(item) for item in parsed]
    elif isinstance(parsed, dict) and isinstance(parsed.get("suggestions"), list):
        candidates = [str(item) for item in parsed["suggestions"]]
    else:
        candidates = [line.strip() for line in content.splitlines() if line.strip()]

    normalized: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        cleaned = _normalize_suggestion_text(item)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized

if TYPE_CHECKING:
    from src.service.rag_service import RAGService
    from src.service.agent_service import AgentService


async def _enrich_citations_with_filename(
    session: AsyncSession, citations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """从 KnowledgeDocument 查询 original_filename，填充 citations 的 filename 字段。"""
    if not citations:
        return citations
    source_ids = list({c["source_id"] for c in citations})
    result = await session.execute(
        select(KnowledgeDocument.source_id, KnowledgeDocument.original_filename).where(
            KnowledgeDocument.source_id.in_(source_ids)
        )
    )
    filename_map = {row.source_id: row.original_filename for row in result.all()}
    for c in citations:
        c["filename"] = filename_map.get(c["source_id"], c.get("filename", c["source_id"]))
    return citations


async def _persist_truncated_assistant_message(
    conversation_id: int,
    content_text: str,
    reasoning_text: str = "",
    citations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """在独立会话中持久化被中断的 assistant 截断消息。"""
    session = AsyncSessionFactory()
    try:
        async with session.begin():
            session.add(
                QaMessage(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=content_text,
                    reasoning_content=reasoning_text or None,
                    citations=citations if citations else None,
                )
            )
    finally:
        await session.close()


class QaService:
    def __init__(
        self,
        session: AsyncSession,
        ai_service: AIService,
        rag_service: Optional["RAGService"] = None,
        agent_service: Optional["AgentService"] = None,
    ):
        self.session = session
        self.ai_service = ai_service
        self.rag_service = rag_service
        self.agent_service = agent_service

    async def create_conversation(self, user_id: int, conversation_data: QaConversationCreate) -> QaConversation:
        """创建新的会话"""
        async with self.session.begin():
            conversation = QaConversation(
                user_id=user_id,
                title=conversation_data.title
            )
            self.session.add(conversation)
            return conversation

    async def send_message(self, conversation_id: int, user_id: int, message_data: QaMessageCreate) -> QaMessage | None:
        async with self.session.begin():
          """发送消息到会话"""
          # 首先验证会话是否属于该用户
          conversation_stmt = select(QaConversation).where(
              QaConversation.id == conversation_id,
              QaConversation.user_id == user_id
          )
          conversation_result = await self.session.execute(conversation_stmt)
          conversation = conversation_result.scalar_one_or_none()

          if not conversation:
              return None

          # 若有图片则保存到本地文件，库中只存 URL（后期可改为 OSS）
          image_url = None
          b64 = getattr(message_data, "image_base64", None) or None
          if b64:
              image_url = save_qa_image_base64_to_file(b64, UPLOAD_DIR)
          # 创建用户消息
          user_message = QaMessage(
              conversation_id=conversation_id,
              role="user",
              content=message_data.content,
              image_url=image_url,
          )
          self.session.add(user_message)

          # 获取对话历史
          history_stmt = select(QaMessage).where(
              QaMessage.conversation_id == conversation_id
          ).order_by(QaMessage.create_time.asc())
          history_result = await self.session.execute(history_stmt)
          history_messages = history_result.scalars().all()

          # 转换为AI服务需要的格式
          preserve_thinking = bool(getattr(message_data, "preserve_thinking", False))
          messages_history = self._build_messages_history(
              history_messages, preserve_thinking=preserve_thinking
          )
          # 根据是否启用 RAG 选择调用方式；RAG 失败（如网络/SSL）时降级为纯 LLM
          model_name = getattr(message_data, "model_name", None)
          model_kwargs = self._build_model_kwargs(message_data)
          use_rag = getattr(message_data, "use_rag", False) and self.rag_service
          citations: List[Dict[str, Any]] = []
          if use_rag:
              logger.info("本次回答使用知识库 (RAG), conversation_id=%s", conversation_id)
              try:
                  ai_response, citations = await self.rag_service.generate_response_with_rag(
                      question=message_data.content,
                      messages=messages_history,
                      system_prompt_prefix="你是一个智能助手，帮助用户解答问题。请尽量基于以下参考内容回答；若参考内容未涉及，可简要说明并建议用户补充。",
                  )
                  ai_response = normalize_citation_format(ai_response)
                  citations = filter_citations_by_referenced(ai_response, citations)
                  citations = await _enrich_citations_with_filename(self.session, citations)
              except Exception as e:
                  logger.warning("知识库检索或调用失败，降级为纯 LLM: %s", e, exc_info=True)
                  ai_response = await self.ai_service.generate_response(
                      messages=messages_history,
                      system_prompt="你是一个智能助手，帮助用户解答问题。",
                      model_kwargs=model_kwargs,
                  )
                  citations = []
          else:
              logger.info("本次回答未使用知识库, conversation_id=%s (use_rag=%s, rag_service=%s)",
                         conversation_id, getattr(message_data, "use_rag", False), self.rag_service is not None)
              ai_response = await self.ai_service.generate_response(
                  messages=messages_history,
                  system_prompt="你是一个智能助手，帮助用户解答问题。",
                  model_name=model_name,
                  model_kwargs=model_kwargs,
              )
          assistant_message = QaMessage(
              conversation_id=conversation_id,
              role="assistant",
              content=ai_response,
              citations=citations if citations else None,
          )
          self.session.add(assistant_message)

          return assistant_message

    async def send_message_stream(
        self,
        conversation_id: int,
        user_id: int,
        message_data: QaMessageCreate,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式发送消息：先落库用户消息，再按 chunk 流式生成 AI 回复并 yield，最后落库 assistant 消息。
        若会话不存在或不属于当前用户，抛出 ValueError，由上层转为 404。
        """
        async with self.session.begin():
            conversation_stmt = select(QaConversation).where(
                QaConversation.id == conversation_id,
                QaConversation.user_id == user_id,
            )
            conversation_result = await self.session.execute(conversation_stmt)
            conversation = conversation_result.scalar_one_or_none()
            if not conversation:
                raise ValueError("会话不存在或不属于该用户")

            image_url = None
            b64 = getattr(message_data, "image_base64", None) or None
            if b64:
                image_url = save_qa_image_base64_to_file(b64, UPLOAD_DIR)
            user_message = QaMessage(
                conversation_id=conversation_id,
                role="user",
                content=message_data.content,
                image_url=image_url,
            )
            self.session.add(user_message)
            await self.session.flush()

            history_stmt = select(QaMessage).where(
                QaMessage.conversation_id == conversation_id
            ).order_by(QaMessage.create_time.asc())
            history_result = await self.session.execute(history_stmt)
            history_messages = history_result.scalars().all()
            preserve_thinking = bool(getattr(message_data, "preserve_thinking", False))
            messages_history = self._build_messages_history(
                history_messages, preserve_thinking=preserve_thinking
            )

        use_rag = getattr(message_data, "use_rag", False) and self.rag_service
        use_image = getattr(message_data, "use_image", False)
        image_base64 = getattr(message_data, "image_base64", None) or ""
        model_name = getattr(message_data, "model_name", None)
        model_kwargs = self._build_model_kwargs(message_data)
        full_content: List[str] = []
        full_reasoning: List[str] = []
        stream_citations: List[Dict[str, Any]] = []
        cancelled = False

        # 调试：流式分支与图像识别条件
        has_agent = self.agent_service is not None
        has_yolo = bool(getattr(self.agent_service, "yolo_service", None) if self.agent_service else False)
        logger.info(
            "[DEBUG] send_message_stream: use_rag=%s, use_image=%s, image_base64_len=%s, agent_service=%s, yolo_service=%s",
            use_rag, use_image, len(image_base64) if image_base64 else 0, has_agent, has_yolo,
        )

        try:
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError
            # 仅在需要知识库或图像识别时才走 Agent 编排，其余场景走纯 LLM 流式回复，保证默认问答有真实流式体验
            use_agent = self.agent_service is not None and (use_rag or use_image)
            if use_agent:
                # Agent 模式：可选注入知识库/图像上下文，再跑 Agent 流式输出
                inject_parts: List[str] = []
                if use_rag and self.rag_service:
                    try:
                        retriever = self.rag_service.knowledge_service.get_retriever()
                        docs = await asyncio.to_thread(
                            retriever.invoke, message_data.content
                        )
                        context, stream_citations = self.rag_service.build_context_and_citations(docs)
                        stream_citations = await _enrich_citations_with_filename(self.session, stream_citations)
                        inject_parts.append(
                            f"【重要】{CITATION_FORMAT_INSTRUCTION}\n\n【知识库参考】\n{context}"
                        )
                        logger.info(
                            "Agent 上下文注入: 知识库, conversation_id=%s",
                            conversation_id,
                        )
                    except Exception as e:
                        logger.exception("知识库注入失败: %s", e)
                        inject_parts.append("【知识库参考】检索失败，请直接回答。")
                        stream_citations = []
                # 图像识别注入：需同时满足 use_image、有 image_base64、yolo_service 已配置
                yolo_svc = getattr(self.agent_service, "yolo_service", None)
                if use_image and image_base64 and yolo_svc:
                    try:
                        logger.info("[DEBUG] 开始调用图像识别, image_base64 长度=%d", len(image_base64))
                        # 规范化 base64：去除空白，补足填充（长度需为 4 的倍数）
                        b64_clean = (image_base64 or "").replace("\n", "").replace("\r", "").strip()
                        pad = (4 - len(b64_clean) % 4) % 4
                        if pad != 4:
                            b64_clean += "=" * pad
                        image_bytes = base64.b64decode(b64_clean, validate=False)
                        detections = await self.agent_service.yolo_service.detect_from_bytes(
                            image_bytes
                        )
                        from src.tools.agent_tools import _format_detections
                        img_text = _format_detections(detections)
                        inject_parts.append(
                            "【用户上传图片识别结果】\n" + img_text
                        )
                        logger.info(
                            "Agent 上下文注入: 图像识别, conversation_id=%s, 检测数=%d",
                            conversation_id, len(detections),
                        )
                    except Exception as e:
                        logger.exception("图像识别注入失败: %s", e)
                        inject_parts.append("【用户上传图片】识别失败，请根据用户文字回答。")
                else:
                    logger.info(
                        "[DEBUG] 未调用图像识别: use_image=%s, image_base64 有值=%s, yolo_service=%s",
                        use_image, bool(image_base64), yolo_svc is not None,
                    )
                current_user_content = message_data.content
                if inject_parts:
                    current_user_content = (
                        "\n\n".join(inject_parts)
                        + "\n\n用户问题：\n"
                        + current_user_content
                    )
                logger.info(
                    "本次流式回答使用 Agent, conversation_id=%s",
                    conversation_id,
                )
                async for chunk in self.agent_service.run_agent_stream(
                    messages_history=messages_history,
                    current_user_content=current_user_content,
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    image_base64=image_base64 if use_image and image_base64 else None,
                ):
                    if cancel_event and cancel_event.is_set():
                        raise asyncio.CancelledError
                    full_content.append(chunk)
                    yield chunk
            elif use_rag:
                logger.info(
                    "本次流式回答使用知识库 (RAG), conversation_id=%s", conversation_id
                )
                ai_response, stream_citations = await self.rag_service.generate_response_with_rag(
                    question=message_data.content,
                    messages=messages_history,
                    system_prompt_prefix="你是一个智能助手，帮助用户解答问题。请尽量基于以下参考内容回答；若参考内容未涉及，可简要说明并建议用户补充。",
                )
                ai_response = normalize_citation_format(ai_response)
                stream_citations = filter_citations_by_referenced(ai_response, stream_citations)
                stream_citations = await _enrich_citations_with_filename(self.session, stream_citations)
                # RAG 目前底层为非流式，这里按固定长度拆分为多个 chunk 提供前端流式体验
                full_content = []
                chunk_size = 50
                for i in range(0, len(ai_response), chunk_size):
                    if cancel_event and cancel_event.is_set():
                        raise asyncio.CancelledError
                    chunk = ai_response[i : i + chunk_size]
                    full_content.append(chunk)
                    yield chunk
            else:
                logger.info(
                    "本次流式回答未使用知识库, conversation_id=%s",
                    conversation_id,
                )
                async for event in self.ai_service.generate_response_stream(
                    messages=messages_history,
                    system_prompt="你是一个智能助手，帮助用户解答问题。",
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                ):
                    if cancel_event and cancel_event.is_set():
                        raise asyncio.CancelledError
                    if event.get("type") == "reasoning_chunk":
                        reasoning_chunk = event.get("content", "")
                        if reasoning_chunk:
                            full_reasoning.append(reasoning_chunk)
                            yield event
                        continue
                    text_chunk = event.get("content", "")
                    if text_chunk:
                        full_content.append(text_chunk)
                        yield event
        except asyncio.CancelledError:
            cancelled = True
            logger.info(
                "流式消息已取消，conversation_id=%s",
                conversation_id,
            )
        finally:
            content_text = "".join(full_content)
            reasoning_text = "".join(full_reasoning)
            content_text = normalize_citation_format(content_text)
            # 只保留回复中实际引用的 citations（<sup>N</sup>）
            stream_citations = filter_citations_by_referenced(content_text, stream_citations)
            is_cancelled = cancelled or (cancel_event and cancel_event.is_set())

            if is_cancelled:
                # 用户中断时：若已生成部分内容，则保存截断结果；不再发送 done 事件
                if content_text.strip():
                    persist_task = asyncio.create_task(
                        _persist_truncated_assistant_message(
                            conversation_id=conversation_id,
                            content_text=content_text,
                            reasoning_text=reasoning_text,
                            citations=stream_citations,
                        )
                    )
                    try:
                        await asyncio.shield(persist_task)
                        logger.info(
                            "流式消息取消后已保存截断 assistant 消息，conversation_id=%s, content_len=%d",
                            conversation_id,
                            len(content_text),
                        )
                    except asyncio.CancelledError:
                        # 请求任务可能已被上游取消，后台任务仍会继续执行持久化
                        logger.warning(
                            "取消写库 await 被中断，截断消息持久化仍在后台继续, conversation_id=%s",
                            conversation_id,
                        )
                    except Exception:
                        logger.exception(
                            "流式取消后保存截断 assistant 消息失败, conversation_id=%s",
                            conversation_id,
                        )
                else:
                    logger.info(
                        "流式消息取消且无可保存内容，conversation_id=%s",
                        conversation_id,
                    )
                return

            # 仅在生成了内容时落库，避免因 DashScope 等异常导致保存空回复
            if content_text.strip():
                assistant_message = QaMessage(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=content_text,
                    reasoning_content=reasoning_text or None,
                    citations=stream_citations if stream_citations else None,
                )
                self.session.add(assistant_message)
                await self.session.commit()
            else:
                logger.warning(
                    "流式回复内容为空，未写入 assistant 消息，conversation_id=%s（可能因网络/API 异常中断）",
                    conversation_id,
                )
            # 流式协议：最后 yield done 事件，携带 citations 供前端 Sources 使用
            yield {"type": "done", "citations": stream_citations}

    @staticmethod
    def _fallback_suggestions(limit: int) -> List[str]:
        return FALLBACK_SUGGESTIONS[:limit]

    async def get_suggestions(
        self,
        conversation_id: int,
        user_id: int,
        message_id: Optional[int] = None,
        limit: int = DEFAULT_SUGGESTION_LIMIT,
    ) -> List[str]:
        capped_limit = max(1, min(limit, MAX_SUGGESTION_LIMIT))

        async with self.session.begin():
            conversation_stmt = select(QaConversation).where(
                QaConversation.id == conversation_id,
                QaConversation.user_id == user_id,
            )
            conversation_result = await self.session.execute(conversation_stmt)
            conversation = conversation_result.scalar_one_or_none()
            if not conversation:
                raise ValueError("会话不存在或不属于该用户")

            if message_id is not None:
                assistant_stmt = (
                    select(QaMessage)
                    .where(
                        QaMessage.id == message_id,
                        QaMessage.conversation_id == conversation_id,
                        QaMessage.role == "assistant",
                    )
                    .limit(1)
                )
            else:
                assistant_stmt = (
                    select(QaMessage)
                    .where(
                        QaMessage.conversation_id == conversation_id,
                        QaMessage.role == "assistant",
                    )
                    .order_by(QaMessage.create_time.desc(), QaMessage.id.desc())
                    .limit(1)
                )
            assistant_result = await self.session.execute(assistant_stmt)
            target_assistant = assistant_result.scalar_one_or_none()

            if not target_assistant or not (target_assistant.content or "").strip():
                return self._fallback_suggestions(capped_limit)

            recent_stmt = (
                select(QaMessage)
                .where(QaMessage.conversation_id == conversation_id)
                .order_by(QaMessage.create_time.desc(), QaMessage.id.desc())
                .limit(8)
            )
            recent_result = await self.session.execute(recent_stmt)
            recent_messages = list(recent_result.scalars().all())

        recent_messages.reverse()
        context_lines: List[str] = []
        for msg in recent_messages:
            text = (msg.content or "").strip()
            if not text:
                continue
            role = "用户" if msg.role == "user" else "助手"
            context_lines.append(f"{role}: {text[:600]}")

        if not context_lines:
            context_lines.append(f"助手: {(target_assistant.content or '').strip()[:600]}")

        prompt = (
            f"基于以下对话上下文，生成 {capped_limit} 个下一轮建议问题。\n"
            "要求：\n"
            "1) 与上下文强相关；\n"
            "2) 句子简短、可直接提问；\n"
            "3) 不要重复，不要解释，不要输出答案；\n"
            "4) 输出 JSON 数组字符串，例如：[\"问题1\",\"问题2\"]。\n\n"
            "对话上下文：\n"
            f"{chr(10).join(context_lines)}"
        )

        try:
            raw = await self.ai_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=(
                    "你是一个追问建议生成器。"
                    "你只负责根据上下文产出可点击的问题建议，"
                    "必须输出 JSON 数组，不要输出其它文本。"
                ),
            )
        except Exception as e:
            logger.warning("生成追问建议失败，使用兜底建议: %s", e, exc_info=True)
            return self._fallback_suggestions(capped_limit)

        suggestions = _extract_suggestions_from_llm(raw)
        if not suggestions:
            return self._fallback_suggestions(capped_limit)
        return suggestions[:capped_limit]

    @staticmethod
    def _build_model_kwargs(message_data: QaMessageCreate) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        enable_thinking = getattr(message_data, "enable_thinking", None)
        thinking_budget = getattr(message_data, "thinking_budget", None)
        preserve_thinking = getattr(message_data, "preserve_thinking", None)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        if thinking_budget is not None:
            kwargs["thinking_budget"] = int(thinking_budget)
        if preserve_thinking is not None:
            kwargs["preserve_thinking"] = bool(preserve_thinking)
        return kwargs

    @staticmethod
    def _build_messages_history(
        history_messages: List[QaMessage], preserve_thinking: bool = False
    ) -> List[Dict[str, str]]:
        messages_history: List[Dict[str, str]] = []
        for msg in history_messages:
            content = msg.content
            # preserve_thinking 打开时，将历史 assistant 的 reasoning 拼接进输入上下文
            if preserve_thinking and msg.role == "assistant" and msg.reasoning_content:
                content = (
                    f"{content}\n\n[assistant_reasoning_context]\n"
                    f"{msg.reasoning_content}\n[/assistant_reasoning_context]"
                )
            messages_history.append({"role": msg.role, "content": content})
        return messages_history

    async def get_conversations(self, user_id: int, skip: int = 0, limit: int = 10) -> tuple[List[QaConversation], int]:
        async with self.session.begin():
          """获取用户的会话列表"""
          # 计算总数
          count_stmt = select(func.count(QaConversation.id)).where(QaConversation.user_id == user_id)
          count_result = await self.session.execute(count_stmt)
          total = count_result.scalar_one()

          # 获取会话列表
          stmt = select(QaConversation).where(
              QaConversation.user_id == user_id
          ).order_by(
              QaConversation.update_time.desc()
          ).offset(skip).limit(limit)

          result = await self.session.execute(stmt)
          conversations = list(result.scalars().all())

          return conversations, total

    async def get_messages(self, conversation_id: int, user_id: int, skip: int = 0, limit: int = 50) -> tuple[List[QaMessage], int]:
        async with self.session.begin():
          """获取会话的消息列表"""
          # 首先验证会话是否属于该用户
          conversation_stmt = select(QaConversation).where(
              QaConversation.id == conversation_id,
              QaConversation.user_id == user_id
          )
          conversation_result = await self.session.execute(conversation_stmt)
          conversation = conversation_result.scalar_one_or_none()

          if not conversation:
              return [], 0

          # 计算总数
          count_stmt = select(func.count(QaMessage.id)).where(QaMessage.conversation_id == conversation_id)
          count_result = await self.session.execute(count_stmt)
          total = count_result.scalar_one()

          # 获取消息列表
          stmt = select(QaMessage).where(
              QaMessage.conversation_id == conversation_id
          ).order_by(
              QaMessage.create_time.asc()
          ).offset(skip).limit(limit)

          result = await self.session.execute(stmt)
          messages = list(result.scalars().all())

          return messages, total
        
    async def delete_conversation(self, conversation_id: int, user_id: int) -> bool:
        async with self.session.begin():
          """删除会话"""
          # 首先验证会话是否属于该用户
          conversation_stmt = select(QaConversation).where(
              QaConversation.id == conversation_id,
              QaConversation.user_id == user_id
          )
          conversation_result = await self.session.execute(conversation_stmt)
          conversation = conversation_result.scalar_one_or_none()

          if not conversation:
              return False

          # 删除会话中的所有消息
          delete_messages_stmt = delete(QaMessage).where(
              QaMessage.conversation_id == conversation_id
          )
          await self.session.execute(delete_messages_stmt)

          # 删除会话
          await self.session.delete(conversation)
          return True

    async def update_conversation_title(
        self, conversation_id: int, user_id: int, title: str
    ) -> bool:
        """更新会话标题，校验归属后更新。"""
        async with self.session.begin():
            stmt = (
                update(QaConversation)
                .where(
                    QaConversation.id == conversation_id,
                    QaConversation.user_id == user_id,
                )
                .values(title=(title or "")[:255])
            )
            result = await self.session.execute(stmt)
            return result.rowcount > 0

    async def generate_conversation_title(
        self, conversation_id: int, user_id: int
    ) -> Optional[str]:
        """
        在第一次对话完成后，根据首条用户问题生成标题并更新。
        仅当消息数为 2 且当前标题为「新对话」时执行。
        """
        async with self.session.begin():
            conversation_stmt = select(QaConversation).where(
                QaConversation.id == conversation_id,
                QaConversation.user_id == user_id,
            )
            conv_result = await self.session.execute(conversation_stmt)
            conversation = conv_result.scalar_one_or_none()
            if not conversation:
                return None

            if (conversation.title or "").strip() != "新对话":
                return conversation.title

            count_stmt = select(func.count(QaMessage.id)).where(
                QaMessage.conversation_id == conversation_id
            )
            count_result = await self.session.execute(count_stmt)
            total = count_result.scalar_one()
            if total != 2:
                return None

            first_user_stmt = (
                select(QaMessage)
                .where(
                    QaMessage.conversation_id == conversation_id,
                    QaMessage.role == "user",
                )
                .order_by(QaMessage.create_time.asc())
                .limit(1)
            )
            msg_result = await self.session.execute(first_user_stmt)
            first_user_msg = msg_result.scalar_one_or_none()
            if not first_user_msg or not (first_user_msg.content or "").strip():
                return None
            user_question = (first_user_msg.content or "").strip()

        try:
            generated = await self.ai_service.generate_short_title(user_question)
        except Exception as e:
            logger.warning("生成对话标题失败: %s", e, exc_info=True)
            return None

        if not generated:
            return None

        success = await self.update_conversation_title(
            conversation_id, user_id, generated
        )
        return generated if success else None