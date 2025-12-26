from sqlalchemy import select, exists
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from schemas.user import UserRegister, UserLogin, UserPasswordChange, UserInfo
from utils.security import hash_password, verify_password

class UserService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register_user(self, user: UserRegister) -> bool:
        async with self.session.begin():
            # 1. 检查用户名是否存在
            existing_user = await self.session.scalar(select(exists().where(User.username == user.username)))
            if existing_user:
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
            return True

    async def login_user(self, login_data: UserLogin) -> User | None:
        async with self.session.begin():
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.username == login_data.username))
            if not user:
                return None
            # 2. 验证密码
            if not verify_password(login_data.password, user.password):
                return None
            return user

    async def get_user_by_username(self, username: str) -> User | None:
        async with self.session.begin():
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.username == username))
            return user


    async def change_password(self, user_data: UserPasswordChange) -> bool:
        async with self.session.begin():
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.username == user_data.username))
            if not user:
                return False
            # 2. 验证密码
            if not verify_password(user_data.password, user.password):
                return False
            # 3. 加密新密码
            hashed_password = hash_password(user_data.new_password)
            # 4. 更新密码
            user.password = hashed_password
            return True

    async def update_user(self, user_data: UserInfo) -> bool:
        async with self.session.begin():
            # 1. 查询用户
            user: User = await self.session.scalar(select(User).where(User.id == user_data.id))
            if not user:
                return False
            # 2. 更新用户信息
            update_fields = {}
            if user_data.real_name is not None:
                user.real_name = user_data.real_name
                update_fields["real_name"] = user_data.real_name
            if user_data.phone is not None:
                user.phone = user_data.phone
                update_fields["phone"] = user_data.phone
            if user_data.email is not None:
                user.email = user_data.email
                update_fields["email"] = user_data.email
            if user_data.role is not None:
                user.role = user_data.role
                update_fields["role"] = user_data.role
            if user_data.status is not None:
                user.status = user_data.status
                update_fields["status"] = user_data.status

            return True

    async def get_user_by_id(self, user_id: int) -> User | None:
        async with self.session.begin():
            # 1. 查询用户
            user = await self.session.scalar(select(User).where(User.id == user_id))
            return user