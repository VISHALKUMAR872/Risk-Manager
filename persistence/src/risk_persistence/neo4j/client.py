from collections.abc import Sequence
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from risk_persistence.config import get_settings


class Neo4jClient:
    def __init__(self) -> None:
        settings = get_settings()

        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(
                settings.neo4j_user,
                settings.neo4j_password,
            ),
        )

    async def verify_connectivity(self) -> None:
        await self.driver.verify_connectivity()

    async def close(self) -> None:
        await self.driver.close()

    async def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        async with self.driver.session() as session:
            result = await session.run(
                query,
                parameters or {},
            )

            return await result.data()