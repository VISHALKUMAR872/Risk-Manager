from collections.abc import AsyncGenerator

from risk_persistence.postgres.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
