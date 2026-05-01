from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import AsyncSessionFactory
from src.models.user import User, DISABLED_STATUS, ADMIN_ROLE
from src.service.user_service import UserService
from src.utils.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")

async def get_session() -> AsyncSession:
    session = AsyncSessionFactory()
    try:
        yield session
    finally:
        await session.close()


async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_session)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # 解码令牌
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    # 查询用户
    user_service = UserService(session)
    user = await user_service.get_user_by_username(username)
    if user is None:
        raise credentials_exception
    
    # 检查用户状态
    if user.status == DISABLED_STATUS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """要求当前用户为管理员。"""
    if current_user.role != ADMIN_ROLE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized",
        )
    return current_user