import asyncio
import base64
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, delete
from typing import List, Optional, TYPE_CHECKING, AsyncGenerator

from src.models.qa import QaConversation, QaMessage

logger = logging.getLogger(__name__)
from src.schemas.qa import QaConversationCreate, QaMessageCreate
from src.service.ai_service import AIService

if TYPE_CHECKING:
    from src.service.rag_service import RAGService
    from src.service.agent_service import AgentService


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

          # 创建用户消息
          user_message = QaMessage(
              conversation_id=conversation_id,
              role="user",
              content=message_data.content
          )
          self.session.add(user_message)

          # 获取对话历史
          history_stmt = select(QaMessage).where(
              QaMessage.conversation_id == conversation_id
          ).order_by(QaMessage.create_time.asc())
          history_result = await self.session.execute(history_stmt)
          history_messages = history_result.scalars().all()

          # 转换为AI服务需要的格式
          messages_history = []
          for msg in history_messages:
              messages_history.append({
                  "role": msg.role,
                  "content": msg.content
              })
          # 根据是否启用 RAG 选择调用方式；RAG 失败（如网络/SSL）时降级为纯 LLM
          use_rag = getattr(message_data, "use_rag", False) and self.rag_service
          if use_rag:
              logger.info("本次回答使用知识库 (RAG), conversation_id=%s", conversation_id)
              try:
                  ai_response = await self.rag_service.generate_response_with_rag(
                      question=message_data.content,
                      messages=messages_history,
                      system_prompt_prefix="你是一个智能助手，帮助用户解答问题。请尽量基于以下参考内容回答；若参考内容未涉及，可简要说明并建议用户补充。",
                  )
              except Exception as e:
                  logger.warning("知识库检索或调用失败，降级为纯 LLM: %s", e, exc_info=True)
                  ai_response = await self.ai_service.generate_response(
                      messages=messages_history,
                      system_prompt="你是一个智能助手，帮助用户解答问题。",
                  )
          else:
              logger.info("本次回答未使用知识库, conversation_id=%s (use_rag=%s, rag_service=%s)",
                         conversation_id, getattr(message_data, "use_rag", False), self.rag_service is not None)
              ai_response = await self.ai_service.generate_response(
                  messages=messages_history,
                  system_prompt="你是一个智能助手，帮助用户解答问题。"
              )
          assistant_message = QaMessage(
              conversation_id=conversation_id,
              role="assistant",
              content=ai_response
          )
          self.session.add(assistant_message)

          return assistant_message

    async def send_message_stream(
        self,
        conversation_id: int,
        user_id: int,
        message_data: QaMessageCreate,
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

            user_message = QaMessage(
                conversation_id=conversation_id,
                role="user",
                content=message_data.content,
            )
            self.session.add(user_message)
            await self.session.flush()

            history_stmt = select(QaMessage).where(
                QaMessage.conversation_id == conversation_id
            ).order_by(QaMessage.create_time.asc())
            history_result = await self.session.execute(history_stmt)
            history_messages = history_result.scalars().all()
            messages_history = [
                {"role": msg.role, "content": msg.content} for msg in history_messages
            ]

        use_rag = getattr(message_data, "use_rag", False) and self.rag_service
        use_image = getattr(message_data, "use_image", False)
        image_base64 = getattr(message_data, "image_base64", None) or ""
        full_content: List[str] = []

        # 调试：流式分支与图像识别条件
        has_agent = self.agent_service is not None
        has_yolo = bool(getattr(self.agent_service, "yolo_service", None) if self.agent_service else False)
        logger.info(
            "[DEBUG] send_message_stream: use_rag=%s, use_image=%s, image_base64_len=%s, agent_service=%s, yolo_service=%s",
            use_rag, use_image, len(image_base64) if image_base64 else 0, has_agent, has_yolo,
        )

        try:
            if self.agent_service is not None:
                # Agent 模式：可选注入知识库/图像上下文，再跑 Agent 流式输出
                inject_parts: List[str] = []
                if use_rag and self.rag_service:
                    try:
                        retriever = self.rag_service.knowledge_service.get_retriever()
                        docs = await asyncio.to_thread(
                            retriever.invoke, message_data.content
                        )
                        context = self.rag_service._build_context_from_docs(docs)
                        inject_parts.append("【知识库参考】\n" + context)
                        logger.info(
                            "Agent 上下文注入: 知识库, conversation_id=%s",
                            conversation_id,
                        )
                    except Exception as e:
                        logger.exception("知识库注入失败: %s", e)
                        inject_parts.append("【知识库参考】检索失败，请直接回答。")
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
                ):
                    full_content.append(chunk)
                    yield chunk
            elif use_rag:
                logger.info(
                    "本次流式回答使用知识库 (RAG), conversation_id=%s", conversation_id
                )
                ai_response = await self.rag_service.generate_response_with_rag(
                    question=message_data.content,
                    messages=messages_history,
                    system_prompt_prefix="你是一个智能助手，帮助用户解答问题。请尽量基于以下参考内容回答；若参考内容未涉及，可简要说明并建议用户补充。",
                )
                full_content = [ai_response]
                yield ai_response
            else:
                logger.info(
                    "本次流式回答未使用知识库, conversation_id=%s",
                    conversation_id,
                )
                async for chunk in self.ai_service.generate_response_stream(
                    messages=messages_history,
                    system_prompt="你是一个智能助手，帮助用户解答问题。",
                ):
                    full_content.append(chunk)
                    yield chunk
        finally:
            content_text = "".join(full_content)
            # 仅在生成了内容时落库，避免因 DashScope 等异常导致保存空回复
            if content_text.strip():
                async with self.session.begin():
                    assistant_message = QaMessage(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=content_text,
                    )
                    self.session.add(assistant_message)
            else:
                logger.warning(
                    "流式回复内容为空，未写入 assistant 消息，conversation_id=%s（可能因网络/API 异常中断）",
                    conversation_id,
                )

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