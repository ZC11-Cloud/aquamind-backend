from pydantic import BaseModel


class UserRegister(BaseModel):
    """用户注册模型"""
    username: str
    password: str
    real_name: str | None = None
    phone: str | None = None
    email: str | None = None
