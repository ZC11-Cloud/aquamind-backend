from sqlalchemy.ext.asyncio import AsyncSession

from models import AsyncSessionFactory


async def get_session() -> AsyncSession:
    session = AsyncSessionFactory()
    try:
        yield session
    finally:
        await session.close()
