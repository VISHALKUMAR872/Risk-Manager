from typing import Any

from risk_engine.contracts.transaction import TransactionEvent

from risk_persistence.neo4j import Neo4jClient


class TransactionGraph:
    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    async def project_transaction(
        self,
        event: TransactionEvent,
    ) -> None:
        query = """
        MERGE (c:Customer {id: $customer_id})
        MERGE (m:Merchant {id: $merchant_id})
        MERGE (d:Device {id: $device_id})
        MERGE (ip:IP {address: $ip_address})
        MERGE (p:PaymentMethod {id: $payment_method_id})

        MERGE (t:Transaction {id: $transaction_id})
        SET
            t.event_id = $event_id,
            t.amount = $amount,
            t.currency = $currency,
            t.event_time = datetime($event_time),
            t.merchant_category = $merchant_category,
            t.country = $country,
            t.channel = $channel

        MERGE (c)-[:MADE]->(t)

        MERGE (t)-[:AT_MERCHANT]->(m)

        MERGE (c)-[used_device:USED_DEVICE]->(d)
        ON CREATE SET used_device.first_seen_at = datetime($event_time)
        ON MATCH SET used_device.first_seen_at =
            CASE
                WHEN used_device.first_seen_at > datetime($event_time)
                THEN datetime($event_time)
                ELSE used_device.first_seen_at
            END

        MERGE (c)-[used_ip:USED_IP]->(ip)
        ON CREATE SET used_ip.first_seen_at = datetime($event_time)
        ON MATCH SET used_ip.first_seen_at =
            CASE
                WHEN used_ip.first_seen_at > datetime($event_time)
                THEN datetime($event_time)
                ELSE used_ip.first_seen_at
            END

        MERGE (c)-[used_payment:USED_PAYMENT]->(p)
        ON CREATE SET used_payment.first_seen_at = datetime($event_time)
        ON MATCH SET used_payment.first_seen_at =
            CASE
                WHEN used_payment.first_seen_at > datetime($event_time)
                THEN datetime($event_time)
                ELSE used_payment.first_seen_at
            END
        """

        await self.client.execute(
            query,
            {
                "customer_id": event.customer_id,
                "merchant_id": event.merchant_id,
                "device_id": event.device_id,
                "ip_address": event.ip_address,
                "payment_method_id": event.payment_method_id,
                "transaction_id": event.transaction_id,
                "event_id": event.event_id,
                "amount": float(event.amount),
                "currency": event.currency,
                "event_time": event.event_time.isoformat(),
                "merchant_category": event.merchant_category,
                "country": event.country,
                "channel": event.channel,
            },
        )

    async def get_transaction_network(
        self,
        transaction_id: str,
        max_related_transactions: int = 8,
    ) -> dict[str, Any] | None:
        """
        Return the selected transaction's entity network plus a bounded
        set of related transactions.

        Related transactions are discovered through shared:
        - customer
        - device
        - IP
        - payment method
        - merchant

        Shared entities are investigation signals only. They do not
        establish that a transaction is fraudulent.
        """

        core_query = """
        MATCH (t:Transaction {id: $transaction_id})

        OPTIONAL MATCH (c:Customer)-[:MADE]->(t)
        OPTIONAL MATCH (t)-[:AT_MERCHANT]->(m:Merchant)

        OPTIONAL MATCH (c)-[:USED_DEVICE]->(d:Device)
        OPTIONAL MATCH (c)-[:USED_IP]->(ip:IP)
        OPTIONAL MATCH (c)-[:USED_PAYMENT]->(p:PaymentMethod)

        RETURN
            t,
            c,
            m,
            d,
            ip,
            p
        """

        rows = await self.client.execute(
            core_query,
            {
                "transaction_id": transaction_id,
            },
        )

        if not rows:
            return None

        row = rows[0]

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        def add_node(
            node: Any,
            node_type: str,
            identifier: str,
            **extra: Any,
        ) -> str | None:
            if node is None:
                return None

            node_id = f"{node_type.lower()}:{identifier}"

            existing = next(
                (
                    item
                    for item in nodes
                    if item["id"] == node_id
                ),
                None,
            )

            if existing is None:
                nodes.append(
                    {
                        "id": node_id,
                        "type": node_type,
                        "label": identifier,
                        **extra,
                    }
                )

            return node_id

        transaction = row["t"]
        customer = row["c"]
        merchant = row["m"]
        device = row["d"]
        ip = row["ip"]
        payment = row["p"]

        if transaction is not None:
            add_node(
                transaction,
                "TRANSACTION",
                transaction["id"],
                selected=True,
                amount=transaction.get("amount"),
                currency=transaction.get("currency"),
                event_time=(
                    str(transaction["event_time"])
                    if transaction.get("event_time") is not None
                    else None
                ),
            )

        customer_node = add_node(
            customer,
            "CUSTOMER",
            customer["id"] if customer is not None else "",
        )

        merchant_node = add_node(
            merchant,
            "MERCHANT",
            merchant["id"] if merchant is not None else "",
        )

        device_node = add_node(
            device,
            "DEVICE",
            device["id"] if device is not None else "",
        )

        ip_node = add_node(
            ip,
            "IP",
            ip["address"] if ip is not None else "",
        )

        payment_node = add_node(
            payment,
            "PAYMENT",
            payment["id"] if payment is not None else "",
        )

        transaction_node = (
            f"transaction:{transaction['id']}"
            if transaction is not None
            else None
        )

        if customer_node and transaction_node:
            edges.append(
                {
                    "source": customer_node,
                    "target": transaction_node,
                    "type": "MADE",
                }
            )

        if transaction_node and merchant_node:
            edges.append(
                {
                    "source": transaction_node,
                    "target": merchant_node,
                    "type": "AT_MERCHANT",
                }
            )

        if customer_node and device_node:
            edges.append(
                {
                    "source": customer_node,
                    "target": device_node,
                    "type": "USED_DEVICE",
                }
            )

        if customer_node and ip_node:
            edges.append(
                {
                    "source": customer_node,
                    "target": ip_node,
                    "type": "USED_IP",
                }
            )

        if customer_node and payment_node:
            edges.append(
                {
                    "source": customer_node,
                    "target": payment_node,
                    "type": "USED_PAYMENT",
                }
            )

        related_query = """
        MATCH (t:Transaction {id: $transaction_id})

        OPTIONAL MATCH (c:Customer)-[:MADE]->(t)
        OPTIONAL MATCH (t)-[:AT_MERCHANT]->(m:Merchant)

        CALL {
            WITH t, c, m

            OPTIONAL MATCH (c)-[:MADE]->(other:Transaction)
            WHERE other.id <> t.id

            RETURN
                other.id AS related_id,
                "CUSTOMER" AS via_type,
                c.id AS via_id,
                "SHARED_CUSTOMER" AS relationship,
                other.amount AS amount,
                other.currency AS currency,
                other.event_time AS event_time

            UNION

            WITH t, c, m

            OPTIONAL MATCH (c)-[:USED_DEVICE]->(d:Device)
            OPTIONAL MATCH (other_customer:Customer)-[:USED_DEVICE]->(d)
            OPTIONAL MATCH (other_customer)-[:MADE]->(other:Transaction)
            WHERE other.id <> t.id

            RETURN
                other.id AS related_id,
                "DEVICE" AS via_type,
                d.id AS via_id,
                "SHARED_DEVICE" AS relationship,
                other.amount AS amount,
                other.currency AS currency,
                other.event_time AS event_time

            UNION

            WITH t, c, m

            OPTIONAL MATCH (c)-[:USED_IP]->(ip:IP)
            OPTIONAL MATCH (other_customer:Customer)-[:USED_IP]->(ip)
            OPTIONAL MATCH (other_customer)-[:MADE]->(other:Transaction)
            WHERE other.id <> t.id

            RETURN
                other.id AS related_id,
                "IP" AS via_type,
                ip.address AS via_id,
                "SHARED_IP" AS relationship,
                other.amount AS amount,
                other.currency AS currency,
                other.event_time AS event_time

            UNION

            WITH t, c, m

            OPTIONAL MATCH (c)-[:USED_PAYMENT]->(p:PaymentMethod)
            OPTIONAL MATCH (other_customer:Customer)-[:USED_PAYMENT]->(p)
            OPTIONAL MATCH (other_customer)-[:MADE]->(other:Transaction)
            WHERE other.id <> t.id

            RETURN
                other.id AS related_id,
                "PAYMENT" AS via_type,
                p.id AS via_id,
                "SHARED_PAYMENT" AS relationship,
                other.amount AS amount,
                other.currency AS currency,
                other.event_time AS event_time

            UNION

            WITH t, c, m

            OPTIONAL MATCH (m)<-[:AT_MERCHANT]-(other:Transaction)
            WHERE other.id <> t.id

            RETURN
                other.id AS related_id,
                "MERCHANT" AS via_type,
                m.id AS via_id,
                "SHARED_MERCHANT" AS relationship,
                other.amount AS amount,
                other.currency AS currency,
                other.event_time AS event_time
        }

        RETURN
            related_id,
            via_type,
            via_id,
            relationship,
            amount,
            currency,
            event_time
        LIMIT $max_related_transactions
        """

        related_rows = await self.client.execute(
            related_query,
            {
                "transaction_id": transaction_id,
                "max_related_transactions": max_related_transactions,
            },
        )

        related_seen: set[str] = set()

        for related in related_rows:
            related_id = related.get("related_id")

            if not related_id:
                continue

            if related_id in related_seen:
                continue

            related_seen.add(related_id)

            event_time = related.get("event_time")

            related_node_id = add_node(
                {"id": related_id},
                "RELATED_TRANSACTION",
                related_id,
                selected=False,
                amount=related.get("amount"),
                currency=related.get("currency"),
                event_time=(
                    str(event_time)
                    if event_time is not None
                    else None
                ),
            )

            via_type = related.get("via_type")
            via_id = related.get("via_id")
            relationship = related.get("relationship")

            if (
                related_node_id
                and via_type
                and via_id
                and relationship
            ):
                via_node_id = (
                    f"{via_type.lower()}:{via_id}"
                )

                if any(
                    node["id"] == via_node_id
                    for node in nodes
                ):
                    edges.append(
                        {
                            "source": via_node_id,
                            "target": related_node_id,
                            "type": relationship,
                        }
                    )

        return {
            "transaction_id": transaction_id,
            "nodes": nodes,
            "edges": edges,
        }
