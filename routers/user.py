import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_session, get_current_user
from models import user
from models.user import User
from schemas.response import ResponseSchema
from schemas.user import UserRegister, UserLogin, Token, UserInfo, UserPasswordChange
from service.user_service import UserService
from settings import ACCESS_TOKEN_EXPIRE_MINUTES
from utils.security import create_access_token

router = APIRouter(prefix="/user", tags=["user"])
logger = logging.getLogger(__name__)
@router.post("/register", response_model=ResponseSchema)
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
    return ResponseSchema(result="success", code=200, message="Register success")


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


@router.post("/password", response_model=ResponseSchema)
async def change_password(user: UserPasswordChange, current_user: UserInfo = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    """修改用户密码"""
    # 1. 用户名密码不能为空
    if not user.username or not user.password or not user.new_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username, password, and new password cannot be empty")
    # 2. 验证用户名
    if user.username != current_user.username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username")
    # 3. 修改密码
    user_service = UserService(session)
    changed = await user_service.change_password(user)
    if not changed:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    return ResponseSchema(result="success", code=200, message="Password changed success")

@router.put("/me", response_model=ResponseSchema)
async def update_user(
        user_update: UserInfo,
        current_user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_session)
):
    """更新当前登录用户信息"""
    # 1. 权限检查： 普通用户只能更新自己，管理员可以更新所有人
    if current_user.role != 1 and user_update.id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update this user")

    # 2. 普通用户不能修改自己的角色和状态
    if current_user.role != 1:
        user_update.role = current_user.role
        user_update.status = current_user.status

    # 3.更新用户信息
    user_service = UserService(session)
    updated = await user_service.update_user(user_update)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return ResponseSchema(result="success", code=200, message="User updated success")

