from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_session, get_current_user
from models import user
from models.user import User
from schemas.response import ResponseSchema
from schemas.user import UserRegister, UserLogin, Token, UserInfo
from service.user_service import UserService
from settings import ACCESS_TOKEN_EXPIRE_MINUTES
from utils.security import create_access_token

router = APIRouter(prefix="/user")

@router.post("/register")
async def register(user: UserRegister, session: AsyncSession = Depends(get_session)):
    """用户注册"""
    # 1. 用户名密码不能为空
    if not user.username or not user.password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password cannot be empty")
    # 2. 注册
    user_service = UserService(session)
    registered = await user_service.register_user(user)
    if not registered:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    return ResponseSchema(message="Register success")


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_session)):
    """用户登录"""
    username = form_data.username
    password = form_data.password
    # 1. 用户名密码不能为空
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password cannot be empty")
    # 2. 登录
    user_service = UserService(session)
    user = await user_service.login_user(UserLogin(username=username, password=password))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    # 3. 创建访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "id": user.id, "role": user.role},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@router.get("/me", response_model=UserInfo)
async def read_users_me(current_user: UserInfo = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return current_user


