from sqlalchemy import Integer, String, DateTime, Enum, text, func
from sqlalchemy.orm import Mapped, mapped_column

from models import Base
from datetime import datetime

class QaConversation(Base):
    __tablename__ = 'qa_conversations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), default=None)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    update_time: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=func.now())

class QaMessage(Base):
    __tablename__ = 'qa_messages'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(Enum('user', 'assistant'), nullable=False)
    content: Mapped[str] = mapped_column(String(255), nullable=False)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'))