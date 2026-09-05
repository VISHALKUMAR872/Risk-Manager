import asyncio

from risk_persistence.neo4j import Neo4jClient


async def main():
    client = Neo4jClient()

    try:
        rows = await client.execute(
            """
            MATCH (c:Customer {id: 'cust-001'})
            MATCH (t:Transaction {id: 'txn-ad8d25b4558f'})
            OPTIONAL MATCH (c)-[d:USED_DEVICE]->(device:Device {id: 'device-001'})
            OPTIONAL MATCH (c)-[ip:USED_IP]->(addr:IP {address: '192.168.1.10'})
            OPTIONAL MATCH (c)-[p:USED_PAYMENT]->(payment:PaymentMethod {id: 'payment-001'})
            RETURN
                t.id AS transaction_id,
                t.event_time AS transaction_event_time,
                d.first_seen_at AS device_first_seen_at,
                ip.first_seen_at AS ip_first_seen_at,
                p.first_seen_at AS payment_first_seen_at
            """
        )

        for row in rows:
            print(row)

    finally:
        await client.close()


asyncio.run(main())
