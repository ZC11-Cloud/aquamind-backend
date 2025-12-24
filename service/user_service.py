from sqlalchemy import select, exists
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from schemas.user import UserRegister
from utils.security import hash_password


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


