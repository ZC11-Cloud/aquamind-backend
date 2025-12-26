from typing import Annotated

from pydantic import BaseModel, Field, EmailStr

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

    class Config:
        from_attributes = True
        use_enum_values = True

class UserPasswordChange(BaseModel):
    """用户密码修改模型"""
    username: UsernameStr
    password: PasswordStr
    new_password: PasswordStr