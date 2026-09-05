import asyncio

from risk_engine.contracts.transaction import TransactionEvent

from risk_persistence.neo4j import Neo4jClient
from risk_persistence.postgres.repositories.transaction_repository import (
    TransactionRepository,
)
from risk_persistence.postgres.session import AsyncSessionLocal
from risk_persistence.neo4j.transaction_graph import TransactionGraph


async def rebuild() -> None:
    neo4j = Neo4jClient()

    try:
        await neo4j.verify_connectivity()

        async with AsyncSessionLocal() as session:
            repository = TransactionRepository(session)
            transactions = await repository.list_all()

        print(f"Loaded {len(transactions)} transactions from PostgreSQL.")

        await neo4j.execute(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )

        print("Cleared Neo4j projection.")

        graph = TransactionGraph(neo4j)

        for index, transaction in enumerate(transactions, start=1):
            event = TransactionEvent(
                event_id=transaction.event_id,
                transaction_id=transaction.transaction_id,
                event_time=transaction.event_time,
                customer_id=transaction.customer_id,
                merchant_id=transaction.merchant_id,
                amount=transaction.amount,
                currency=transaction.currency,
                device_id=transaction.device_id,
                ip_address=transaction.ip_address,
                payment_method_id=transaction.payment_method_id,
                merchant_category=transaction.merchant_category,
                country=transaction.country,
                channel=transaction.channel,
            )

            await graph.project_transaction(event)

            print(
                f"[{index}/{len(transactions)}] "
                f"projected {transaction.transaction_id}"
            )

        print("Neo4j projection rebuild completed.")

    finally:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(rebuild())