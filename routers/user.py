from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_session
from schemas.response import ResponseSchema
from schemas.user import UserRegister
from service.user_service import UserService

router = APIRouter(prefix="/user")

@router.post("/register")
async def register(user: UserRegister, session: AsyncSession = Depends(get_session)):
    """用户注册"""
    # 1. 用户名密码不能为空
    if not user.username or not user.password:
        raise HTTPException(status_code=400, detail="Username and password cannot be empty")
    # 2. 注册
    user_service = UserService(session)
    registered = await user_service.register_user(user)
    if not registered:
        raise HTTPException(status_code=400, detail="Username already exists")
    return ResponseSchema(message="Register success")

