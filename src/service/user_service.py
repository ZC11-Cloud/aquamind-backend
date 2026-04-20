import logging
from sqlalchemy import select, exists
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.schemas.user import (
    UserRegister,
    UserLogin,
    UserPasswordChange,
    UserInfo,
    UserProfileUpdate,
)
from src.utils.security import hash_password, verify_password
logger = logging.getLogger(__name__)
class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register_user(self, user: UserRegister) -> bool:
        async with self.session.begin():
            logger.info("register_user: 尝试注册 username=%s", user.username)
            # 1. 检查用户名是否存在
            existing_user = await self.session.scalar(select(exists().where(User.username == user.username)))
            if existing_user:
                logger.info("register_user: 用户名已存在 username=%s", user.username)
                return False
            # 2 .密码加密
            hashed_password = hash_password(user.password)
            user = User(
                username=user.username,
                password=hashed_password,
                real_name=user.real_name,
                phone=user.phone,
                email=user.email
            )
            # 3. 创建用户
            self.session.add(user)
            logger.info("register_user: 注册成功 username=%s", user.username)
            return True

    async def login_user(self, login_data: UserLogin) -> User | None:
        async with self.session.begin():
            pwd_len = len(login_data.password.encode("utf-8"))
            logger.info("login_user: 尝试登录 username=%s, 密码长度=%d 字节", login_data.username, pwd_len)
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.username == login_data.username))
            if not user:
                logger.info("login_user: 用户不存在 username=%s", login_data.username)
                return None
            # 2. 验证密码
            if not verify_password(login_data.password, user.password):
                logger.info("login_user: 密码错误 username=%s", login_data.username)
                return None
            logger.info("login_user: 登录成功 username=%s, user_id=%s", login_data.username, user.id)
            return user

    async def get_user_by_username(self, username: str) -> User | None:
        async with self.session.begin():
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.username == username))
            return user

    async def change_password(self, user_data: UserPasswordChange) -> bool:
        async with self.session.begin():
            logger.info("change_password: 尝试修改密码 username=%s", user_data.username)
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.username == user_data.username))
            if not user:
                logger.warning("change_password: 用户不存在 username=%s", user_data.username)
                return False
            # 2. 验证密码
            if not verify_password(user_data.password, user.password):
                logger.info("change_password: 原密码错误 username=%s", user_data.username)
                return False
            # 3. 加密新密码
            hashed_password = hash_password(user_data.new_password)
            # 4. 更新密码
            user.password = hashed_password
            logger.info("change_password: 修改成功 username=%s", user_data.username)
            return True

    async def update_user_profile(self, user_id: int, user_data: UserProfileUpdate) -> bool:
        async with self.session.begin():
            # 1. 查询用户
            user: User = await self.session.scalar(select(User).where(User.id == user_id))
            if not user:
                return False
            # 2. 更新用户信息
            if user_data.real_name is not None:
                user.real_name = user_data.real_name
            if user_data.phone is not None:
                user.phone = user_data.phone
            if user_data.email is not None:
                user.email = user_data.email

            return True

    async def get_user_by_id(self, user_id: int) -> User | None:
        async with self.session.begin():
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.id == user_id))
            return user

    async def update_user_avatar(self, user_id: int, bucket: str, object_key: str) -> bool:
        """更新用户头像"""
        async with self.session.begin():
            # 1. 查询用户
            user: User = await self.session.scalar(select(User).where(User.id == user_id))
            if not user:
                return False
            # 2. 更新用户头像
            user.avatar_bucket = bucket
            user.avatar_object_key = object_key
            return True