import logging
import os
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies import get_session, get_current_user, require_admin
from src.models.user import User, ADMIN_ROLE, NORMAL_ROLE, DISABLED_STATUS
from src.schemas.response import ResponseSchema
from src.schemas.user import (
    UserRegister,
    UserLogin,
    Token,
    UserInfo,
    UserPasswordChange,
    UserProfileUpdate,
    AdminUserCreate,
    AdminUserUpdate,
    AdminUserStatusUpdate,
    AdminUserRoleUpdate,
    AdminUserListItem,
)
from src.service.user_service import UserService
from src.settings import ACCESS_TOKEN_EXPIRE_MINUTES, UPLOAD_DIR
from src.utils.security import create_access_token

router = APIRouter(prefix="/user", tags=["user"])
logger = logging.getLogger(__name__)


def _build_user_info(user_obj: User) -> UserInfo:
    user_info = UserInfo.model_validate(user_obj)
    if user_obj.avatar_bucket and user_obj.avatar_object_key:
        user_info.avatar_url = f"/user/avatar/{user_obj.id}"
    return user_info


def _build_admin_user_item(user_obj: User) -> AdminUserListItem:
    user_info = AdminUserListItem.model_validate(user_obj)
    if user_obj.avatar_bucket and user_obj.avatar_object_key:
        user_info.avatar_url = f"/user/avatar/{user_obj.id}"
    return user_info


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
    pwd_bytes = len(password.encode("utf-8"))
    logger.info("login: 收到登录请求 username=%s, 密码长度=%d 字节", username, pwd_bytes)
    # 1. 用户名密码不能为空
    if not username or not password:
        logger.warning("login: 用户名或密码为空")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password cannot be empty")
    # 2. 登录
    user_service = UserService(session)
    user = await user_service.login_user(UserLogin(username=username, password=password))
    if not user:
        logger.warning("login: 登录失败 username=%s", username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    logger.info("login: 登录成功 username=%s", username)
    # 3. 创建访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "id": user.id, "role": user.role},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@router.get("/me", response_model=ResponseSchema)
async def read_users_me(
        current_user: User = Depends(get_current_user)
):
    """获取当前登录用户信息"""
    user_info = _build_user_info(current_user)
    return ResponseSchema(
        result="success",
        code=200,
        message="User info retrieved success",
        data=user_info.model_dump(),
    )


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
        user_update: UserProfileUpdate,
        current_user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_session)
):
    """更新当前登录用户信息"""
    # 1. 更新用户信息
    user_service = UserService(session)
    updated = await user_service.update_user_profile(current_user.id, user_update)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    refreshed_user = await user_service.get_user_by_id(current_user.id)
    if not refreshed_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_info = _build_user_info(refreshed_user)
    return ResponseSchema(
        result="success",
        code=200,
        message="User updated success",
        data=user_info.model_dump(),
    )


@router.get("/admin/list", response_model=ResponseSchema)
async def list_users_by_admin(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=100),
        keyword: str | None = Query(None, max_length=50),
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """管理员分页获取用户列表。"""
    _ = current_user
    user_service = UserService(session)
    users, total = await user_service.list_users(page=page, page_size=page_size, keyword=keyword)
    items = [_build_admin_user_item(user).model_dump() for user in users]
    return ResponseSchema(
        result="success",
        code=200,
        message="User list retrieved success",
        data={
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    )


@router.get("/admin/{user_id}", response_model=ResponseSchema)
async def get_user_info_by_admin(
        user_id: int,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """管理员获取指定用户详情。"""
    _ = current_user
    user_service = UserService(session)
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_info = _build_admin_user_item(user)
    return ResponseSchema(
        result="success",
        code=200,
        message="User info retrieved success",
        data=user_info.model_dump(),
    )


@router.post("/admin", response_model=ResponseSchema)
async def add_user_by_admin(
        user: AdminUserCreate,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """管理员新增用户。"""
    _ = current_user
    user_service = UserService(session)
    added = await user_service.create_user_by_admin(user)
    if not added:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    return ResponseSchema(result="success", code=200, message="User added success")


@router.put("/admin/{user_id}", response_model=ResponseSchema)
async def update_user_by_admin(
        user_id: int,
        user_update: AdminUserUpdate,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """管理员更新用户基础信息。"""
    _ = current_user
    user_service = UserService(session)
    updated_user = await user_service.update_user_by_admin(user_id=user_id, user_data=user_update)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_info = _build_admin_user_item(updated_user)
    return ResponseSchema(
        result="success",
        code=200,
        message="User updated success",
        data=user_info.model_dump(),
    )


@router.patch("/admin/{user_id}/status", response_model=ResponseSchema)
async def update_user_status_by_admin(
        user_id: int,
        payload: AdminUserStatusUpdate,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """管理员更新用户状态。"""
    user_service = UserService(session)
    target_user = await user_service.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target_user.id == current_user.id and payload.status == DISABLED_STATUS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot disable current admin account")

    if target_user.role == ADMIN_ROLE and payload.status == DISABLED_STATUS:
        admin_count = await user_service.count_admin_users()
        if admin_count <= 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot disable the last admin")

    updated_user = await user_service.update_user_status(user_id=user_id, status_value=payload.status)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_info = _build_admin_user_item(updated_user)
    return ResponseSchema(
        result="success",
        code=200,
        message="User status updated success",
        data=user_info.model_dump(),
    )


@router.patch("/admin/{user_id}/role", response_model=ResponseSchema)
async def update_user_role_by_admin(
        user_id: int,
        payload: AdminUserRoleUpdate,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """管理员更新用户角色。"""
    user_service = UserService(session)
    target_user = await user_service.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target_user.id == current_user.id and payload.role == NORMAL_ROLE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot demote current admin account")

    if target_user.role == ADMIN_ROLE and payload.role == NORMAL_ROLE:
        admin_count = await user_service.count_admin_users()
        if admin_count <= 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot demote the last admin")

    updated_user = await user_service.update_user_role(user_id=user_id, role_value=payload.role)
    if not updated_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_info = _build_admin_user_item(updated_user)
    return ResponseSchema(
        result="success",
        code=200,
        message="User role updated success",
        data=user_info.model_dump(),
    )


@router.delete("/admin/{user_id}", response_model=ResponseSchema)
async def delete_user_by_admin(
        user_id: int,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """管理员删除用户。"""
    user_service = UserService(session)
    target_user = await user_service.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target_user.id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete current admin account")

    if target_user.role == ADMIN_ROLE:
        admin_count = await user_service.count_admin_users()
        if admin_count <= 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete the last admin")

    deleted = await user_service.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return ResponseSchema(result="success", code=200, message="User deleted success")


@router.get("/{user_id}", response_model=ResponseSchema)
async def get_user_info(
        user_id: int,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """兼容旧版路径：管理员获取指定用户详细信息。"""
    _ = current_user
    user_service = UserService(session)
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_info = _build_user_info(user)
    return ResponseSchema(result="success", code=200, message="User info retrieved success", data=user_info.model_dump())


@router.post("", response_model=ResponseSchema)
async def add_user(
        user: UserRegister,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_session)
):
    """兼容旧版路径：管理员新增普通用户。"""
    _ = current_user
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

    # 本地存储（与 main.py 静态目录一致，使用统一配置 UPLOAD_DIR）
    avatar_dir = os.path.join(UPLOAD_DIR, "avatars")
    os.makedirs(avatar_dir, exist_ok=True)
    file_path = os.path.join(avatar_dir, unique_filename)
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
    if current_user.role != ADMIN_ROLE and user_id != current_user.id:
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