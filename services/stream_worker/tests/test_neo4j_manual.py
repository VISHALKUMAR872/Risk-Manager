import asyncio

from risk_persistence.neo4j import Neo4jClient


async def main() -> None:
    neo4j = Neo4jClient()

    await neo4j.verify_connectivity()

    result = await neo4j.execute(
        "RETURN 1 AS value"
    )

    print(result)

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())