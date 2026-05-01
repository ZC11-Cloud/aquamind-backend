import logging
import asyncio
from sqlalchemy import select, exists, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User, NORMAL_ROLE, ACTIVE_STATUS, ADMIN_ROLE
from src.schemas.user import (
    UserRegister,
    UserLogin,
    UserPasswordChange,
    UserProfileUpdate,
    AdminUserCreate,
    AdminUserUpdate,
)
from src.utils.security import hash_password, verify_password
logger = logging.getLogger(__name__)
class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register_user(self, user: UserRegister) -> bool:
        return await self.create_user(
            user_data=user,
            role=NORMAL_ROLE,
            status=ACTIVE_STATUS,
        )

    async def create_user(self, user_data: UserRegister, role: int, status: int) -> bool:
        async with self.session.begin():
            logger.info("register_user: 尝试注册 username=%s", user_data.username)
            # 1. 检查用户名是否存在
            existing_user = await self.session.scalar(
                select(exists().where(User.username == user_data.username))
            )
            if existing_user:
                logger.info("register_user: 用户名已存在 username=%s", user_data.username)
                return False
            # 2 .密码加密
            hashed_password = await asyncio.to_thread(hash_password, user_data.password)
            user = User(
                username=user_data.username,
                password=hashed_password,
                real_name=user_data.real_name,
                phone=user_data.phone,
                email=user_data.email,
                role=role,
                status=status,
            )
            # 3. 创建用户
            self.session.add(user)
            logger.info("register_user: 注册成功 username=%s", user.username)
            return True

    async def create_user_by_admin(self, user_data: AdminUserCreate) -> bool:
        return await self.create_user(
            user_data=user_data,
            role=user_data.role,
            status=user_data.status,
        )

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
            is_valid = await asyncio.to_thread(
                verify_password, login_data.password, user.password
            )
            if not is_valid:
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
            is_valid = await asyncio.to_thread(
                verify_password, user_data.password, user.password
            )
            if not is_valid:
                logger.info("change_password: 原密码错误 username=%s", user_data.username)
                return False
            # 3. 加密新密码
            hashed_password = await asyncio.to_thread(
                hash_password, user_data.new_password
            )
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

    async def list_users(self, page: int, page_size: int, keyword: str | None) -> tuple[list[User], int]:
        async with self.session.begin():
            filter_condition = None
            if keyword:
                clean_keyword = keyword.strip()
                if clean_keyword:
                    pattern = f"%{clean_keyword}%"
                    filter_condition = or_(
                        User.username.ilike(pattern),
                        User.real_name.ilike(pattern),
                        User.email.ilike(pattern),
                        User.phone.ilike(pattern),
                    )

            count_query = select(func.count()).select_from(User)
            list_query = select(User)
            if filter_condition is not None:
                count_query = count_query.where(filter_condition)
                list_query = list_query.where(filter_condition)

            total_result = await self.session.execute(count_query)
            total = total_result.scalar_one()

            offset = (page - 1) * page_size
            users_result = await self.session.execute(
                list_query.order_by(User.create_time.desc()).offset(offset).limit(page_size)
            )
            users = users_result.scalars().all()
            return users, total

    async def update_user_by_admin(self, user_id: int, user_data: AdminUserUpdate) -> User | None:
        async with self.session.begin():
            user = await self.session.scalar(select(User).where(User.id == user_id))
            if not user:
                return None
            if user_data.real_name is not None:
                user.real_name = user_data.real_name
            if user_data.phone is not None:
                user.phone = user_data.phone
            if user_data.email is not None:
                user.email = user_data.email
            return user

    async def update_user_status(self, user_id: int, status_value: int) -> User | None:
        async with self.session.begin():
            user = await self.session.scalar(select(User).where(User.id == user_id))
            if not user:
                return None
            user.status = status_value
            return user

    async def update_user_role(self, user_id: int, role_value: int) -> User | None:
        async with self.session.begin():
            user = await self.session.scalar(select(User).where(User.id == user_id))
            if not user:
                return None
            user.role = role_value
            return user

    async def delete_user(self, user_id: int) -> bool:
        async with self.session.begin():
            user = await self.session.scalar(select(User).where(User.id == user_id))
            if not user:
                return False
            await self.session.delete(user)
            return True

    async def count_admin_users(self) -> int:
        async with self.session.begin():
            result = await self.session.execute(
                select(func.count()).select_from(User).where(User.role == ADMIN_ROLE)
            )
            return result.scalar_one()