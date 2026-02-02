from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from typing import List, Optional

from models.qa import QaConversation, QaMessage
from schemas.qa import QaConversationCreate, QaMessageCreate


class QaService:
    def __init__(self, session: AsyncSession):
        self.session = session

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

          # 这里应该添加调用AI生成回复的逻辑
          # 为了演示，我们创建一个简单的模拟回复
          assistant_message = QaMessage(
              conversation_id=conversation_id,
              role="assistant",
              content=f"我收到了你的消息：{message_data.content}"
          )
          self.session.add(assistant_message)

          return assistant_message

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