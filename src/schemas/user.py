from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, EmailStr
from src.models.user import NORMAL_ROLE, ACTIVE_STATUS

UsernameStr = Annotated[str, Field(..., min_length=3, max_length=50)]
PasswordStr = Annotated[str, Field(..., min_length=6, max_length=100)]
PhoneStr = Annotated[str | None, Field(None, pattern=r'^1[3-9]\d{9}$')]

class UserRegister(BaseModel):
    """用户注册模型"""
    username: UsernameStr
    password: PasswordStr
    real_name: str | None = None
    phone: PhoneStr
    email: EmailStr | None = None

class UserLogin(BaseModel):
    """用户登录模型"""
    username: UsernameStr
    password: PasswordStr


class Token(BaseModel):
    """令牌模型"""
    access_token: str
    token_type: str


class UserInfo(BaseModel):
    """用户信息模型"""
    id: int
    username: UsernameStr
    real_name: str | None = None
    phone: PhoneStr
    email: EmailStr | None = None
    role: int
    status: int
    avatar_url: str | None = None
    class Config:
        from_attributes = True
        use_enum_values = True

class UserProfileUpdate(BaseModel):
    """用户可更新的个人资料字段"""
    real_name: str | None = None
    phone: PhoneStr
    email: EmailStr | None = None

class UserPasswordChange(BaseModel):
    """用户密码修改模型"""
    username: UsernameStr
    password: PasswordStr
    new_password: PasswordStr


class AdminUserCreate(UserRegister):
    """管理员创建用户模型。"""
    role: Literal[0, 1] = NORMAL_ROLE
    status: Literal[0, 1] = ACTIVE_STATUS


class AdminUserUpdate(BaseModel):
    """管理员更新用户基础资料模型。"""
    real_name: str | None = None
    phone: PhoneStr
    email: EmailStr | None = None


class AdminUserStatusUpdate(BaseModel):
    """管理员更新用户状态模型。"""
    status: Literal[0, 1]


class AdminUserRoleUpdate(BaseModel):
    """管理员更新用户角色模型。"""
    role: Literal[0, 1]


class AdminUserListItem(UserInfo):
    """管理员端用户列表项。"""
    create_time: datetime
    update_time: datetime