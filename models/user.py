from datetime import datetime

from sqlalchemy import Integer, String, DateTime, text, func
from sqlalchemy.orm import Mapped, mapped_column

from models import Base


class User(Base):
    __tablename__ = 'users'

    DEFAULT_ROLE = 0
    DEFAULT_STATUS = 1

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(100), nullable=False)
    real_name: Mapped[str | None] = mapped_column(String(50), default=None)
    phone: Mapped[str | None] = mapped_column(String(11), default=None)
    email: Mapped[str | None] = mapped_column(String(100), default=None)
    avatar_bucket: Mapped[str | None] = mapped_column(String(50), default=None)
    avatar_object_key: Mapped[str | None] = mapped_column(String(255), default=None)
    role: Mapped[int] = mapped_column(Integer, nullable=False, default=DEFAULT_ROLE)
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=DEFAULT_STATUS)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    update_time: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'),
                                                  onupdate=func.now())