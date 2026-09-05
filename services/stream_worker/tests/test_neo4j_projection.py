import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from risk_engine.contracts.transaction import TransactionEvent
from risk_persistence.neo4j import Neo4jClient, TransactionGraph


async def main() -> None:
    event = TransactionEvent(
        event_id="evt-neo4j-test",
        transaction_id="txn-neo4j-test",
        event_time=datetime.now(timezone.utc),
        customer_id="cust-neo4j-test",
        merchant_id="merchant-neo4j-test",
        amount=Decimal("1499.00"),
        currency="INR",
        device_id="device-neo4j-test",
        ip_address="10.10.10.10",
        payment_method_id="payment-neo4j-test",
        merchant_category="electronics",
        country="IN",
        channel="web",
    )

    client = Neo4jClient()

    await client.verify_connectivity()

    graph = TransactionGraph(client)

    await graph.project_transaction(event)

    print("Neo4j projection complete.")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())