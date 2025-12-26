from datetime import datetime

from sqlalchemy import Integer, String, DateTime, text, func
from sqlalchemy.orm import Mapped, mapped_column

from models import Base

from enum import Enum

class UserRole(int, Enum):
    """用户角色枚举"""
    NORMAL = 0  # 普通用户
    ADMIN = 1   # 管理员

class UserStatus(int, Enum):
    """用户状态枚举"""
    DISABLED = 0  # 禁用
    ACTIVE = 1    # 活跃

class User(Base):
    __tablename__ = 'users'

    DEFAULT_ROLE = UserRole.NORMAL
    DEFAULT_STATUS = UserStatus.ACTIVE

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(100), nullable=False)
    real_name: Mapped[str | None] = mapped_column(String(50), default=None)
    phone: Mapped[str | None] = mapped_column(String(11), default=None)
    email: Mapped[str | None] = mapped_column(String(100), default=None)
    avatar_bucket: Mapped[str | None] = mapped_column(String(50), default=None)
    avatar_object_key: Mapped[str | None] = mapped_column(String(255), default=None)
    role: Mapped[UserRole] = mapped_column(Integer, nullable=False, default=DEFAULT_ROLE)
    status: Mapped[UserStatus] = mapped_column(Integer, nullable=False, default=DEFAULT_STATUS)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    update_time: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'),
                                                  onupdate=func.now())