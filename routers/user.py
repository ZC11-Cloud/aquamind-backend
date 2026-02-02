import logging
import os
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
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
async def register(
        user: UserRegister,
        session: AsyncSession = Depends(get_session)
):
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
async def login(
        form_data: OAuth2PasswordRequestForm = Depends(),
        session: AsyncSession = Depends(get_session)
):
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
async def read_users_me(
        current_user: UserInfo = Depends(get_current_user)
):
    """获取当前登录用户信息"""
    return current_user


@router.post("/password", response_model=ResponseSchema)
async def change_password(
        user: UserPasswordChange,
        current_user: UserInfo = Depends(get_current_user),
        session: AsyncSession = Depends(get_session)
):
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


@router.get("/{user_id}", response_model=ResponseSchema)
async def get_user_info(
        user_id: int,
        current_user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_session)
):
    """获取指定用户详细信息（管理员权限）"""
    # 1. 权限检查： 只有管理员可以查询其他用户信息
    if current_user.role != 1:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this user")
    # 2. 查询用户
    user_service = UserService(session)
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_info = UserInfo.model_validate(user)
    return ResponseSchema(result="success", code=200, message="User info retrieved success", data=user_info.model_dump())


@router.post("", response_model=ResponseSchema)
async def add_user(
        user: UserRegister,
        current_user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_session)
):
    """添加新用户（管理员权限）"""
    # 1. 权限检查： 只有管理员可以添加用户
    if current_user.role != 1:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to add user")
    # 2. 添加用户
    user_service = UserService(session)
    added = await user_service.register_user(user)
    if not added:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    return ResponseSchema(result="success", code=200, message="User added success")


@router.post("/avatar", response_model=ResponseSchema)
async def upload_avatar(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_session)
):
    """上传用户头像"""
    # 1. 验证文件类型
    allowed_extensions = {"png", "jpg", "jpeg", "gif"}
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing"
        )
    file_extension = os.path.splitext(file.filename)[1][1:].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only png, jpg, and gif files are allowed"
        )
    # 2. 验证文件大小
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds 5MB limit"
        )
    # 3.生成唯一的文件名
    unique_filename = f"{current_user.id}_{uuid.uuid4()}.{file_extension}"

    # 本地存储
    UPLOAD_DIR = "uploads/avatars"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open (file_path, "wb") as f:
        f.write(contents)

    bucket = "local"
    object_key = file_path

    # 4. 更新用户头像信息
    user_service = UserService(session)
    updated = await user_service.update_user_avatar(current_user.id, bucket, object_key)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return ResponseSchema(
        result="success",
        code=200,
        message="Avatar updated success"
    )

@router.get("/avatar/{user_id}", response_model=ResponseSchema)
async def get_avatar(
        user_id: int,
        current_user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_session)
):
    """获取用户头像"""
    # 1. 权限检查： 普通用户只能查询自己，管理员可以查询所有用户
    if current_user.role != 1 and user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user avatar")
    # 2. 查询用户头像
    user_service = UserService(session)
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # 3. 根据存储方式返回图像
    if user.avatar_bucket == "local" and user.avatar_object_key:
        # 本地存储： 返回文件内容
        from fastapi.responses import FileResponse
        return FileResponse(user.avatar_object_key)
    elif user.avatar_bucket and user.avatar_object_key:
        # 云存储： 返回重定向URL或直接返回文件
        pass

    # 4. 如果没有头像，返回默认头像
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User avatar not found"
    )