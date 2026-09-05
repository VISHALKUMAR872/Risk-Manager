from __future__ import annotations

import asyncio
import sys
from decimal import Decimal

from sqlalchemy import select, text

from risk_persistence.postgres.models.transaction import Transaction
from risk_persistence.postgres.models.risk_decision import RiskDecision
from risk_persistence.postgres.session import AsyncSessionLocal
from risk_persistence.redis import RedisClient
from risk_persistence.neo4j import Neo4jClient


EXPECTED_AMOUNT = Decimal("1499.00")
EXPECTED_IP = "10.250.250.10"

EXPECTED_MODEL = "fraud-online-v5"
EXPECTED_CALIBRATION = "isotonic-online-v5"
EXPECTED_POLICY = "policy-v5-balanced"

EXPECTED_DECISION = "VERIFY"
EXPECTED_RISK_LEVEL = "MEDIUM"
EXPECTED_REASON = "ELEVATED_EXPECTED_LOSS"


def check(label: str, condition: bool, detail: str) -> None:
    if condition:
        print(f"[PASS] {label}: {detail}")
    else:
        print(f"[FAIL] {label}: {detail}")
        raise AssertionError(f"{label}: {detail}")


async def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python verify_v5_e2e.py <transaction_id>"
        )

    transaction_id = sys.argv[1]

    redis = RedisClient()
    neo4j = Neo4jClient()

    try:
        print("=" * 88)
        print("RISK SENTINEL — V5 LIVE E2E PERSISTENCE VERIFICATION")
        print("=" * 88)
        print(f"Transaction: {transaction_id}")

        # ================================================================
        # 1. PostgreSQL transaction
        # ================================================================

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Transaction).where(
                    Transaction.transaction_id == transaction_id
                )
            )

            transaction = result.scalar_one_or_none()

            check(
                "PostgreSQL transaction exists",
                transaction is not None,
                transaction_id,
            )

            assert transaction is not None

            event_id = transaction.event_id
            customer_id = transaction.customer_id
            device_id = transaction.device_id
            ip_address = transaction.ip_address

            check(
                "Event ID",
                bool(event_id),
                event_id,
            )

            check(
                "Transaction status",
                transaction.status == "DECIDED",
                transaction.status,
            )

            check(
                "Amount",
                transaction.amount == EXPECTED_AMOUNT,
                str(transaction.amount),
            )

            check(
                "Customer ID",
                bool(customer_id),
                customer_id,
            )

            check(
                "Device ID",
                bool(device_id),
                device_id,
            )

            check(
                "IP address",
                ip_address == EXPECTED_IP,
                ip_address,
            )

        # ================================================================
        # 2. PostgreSQL risk decision
        # ================================================================

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RiskDecision).where(
                    RiskDecision.transaction_id == transaction_id
                )
            )

            decision = result.scalar_one_or_none()

            check(
                "Risk decision exists",
                decision is not None,
                transaction_id,
            )

            assert decision is not None

            check(
                "Decision",
                decision.decision == EXPECTED_DECISION,
                decision.decision,
            )

            check(
                "Risk level",
                decision.risk_level == EXPECTED_RISK_LEVEL,
                decision.risk_level,
            )

            check(
                "Reason code",
                EXPECTED_REASON in decision.reason_codes,
                str(decision.reason_codes),
            )

            check(
                "Model version",
                decision.model_version == EXPECTED_MODEL,
                decision.model_version,
            )

            check(
                "Calibration version",
                decision.calibration_version == EXPECTED_CALIBRATION,
                decision.calibration_version,
            )

            check(
                "Policy version",
                decision.policy_version == EXPECTED_POLICY,
                decision.policy_version,
            )

            check(
                "Calibrated probability",
                0.0 <= float(decision.fraud_probability) <= 1.0,
                f"{decision.fraud_probability:.10f}",
            )

            check(
                "Expected loss",
                float(decision.expected_loss) >= 0.0,
                f"{decision.expected_loss:.4f} INR",
            )

            actual_probability = float(decision.fraud_probability)
            actual_expected_loss = float(decision.expected_loss)

        # ================================================================
        # 3. PostgreSQL outbox
        # ================================================================

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    """
                    SELECT
                        event_type,
                        aggregate_type,
                        aggregate_id,
                        topic,
                        message_key,
                        payload
                    FROM outbox_events
                    WHERE aggregate_id = :transaction_id
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
                {
                    "transaction_id": transaction_id
                },
            )

            outbox = result.mappings().first()

            check(
                "Outbox event exists",
                outbox is not None,
                transaction_id,
            )

            assert outbox is not None

            check(
                "Outbox event type",
                outbox["event_type"] == "RiskDecisionCreated",
                str(outbox["event_type"]),
            )

            check(
                "Outbox aggregate ID",
                outbox["aggregate_id"] == transaction_id,
                str(outbox["aggregate_id"]),
            )

            check(
                "Outbox message key",
                outbox["message_key"] == transaction_id,
                str(outbox["message_key"]),
            )

        # ================================================================
        # 4. Redis
        # ================================================================

        await redis.ping()

        customer_key = f"customer:{customer_id}:tx_events"
        device_key = f"device:{device_id}:tx_events"
        ip_key = f"ip:{ip_address}:tx_events"
        idempotency_key = f"velocity:processed:{transaction_id}"

        print("\nRedis state:")

        for label, key in [
            ("Customer velocity", customer_key),
            ("Device velocity", device_key),
            ("IP velocity", ip_key),
        ]:
            members = await redis.client.zrange(
                key,
                0,
                -1,
                withscores=True,
            )

            print(f"  {label:<20}: {members}")

            check(
                label,
                any(
                    member == transaction_id
                    for member, _ in members
                ),
                f"transaction present in {key}",
            )

        marker = await redis.get(idempotency_key)

        check(
            "Redis idempotency marker",
            marker == "1",
            f"{idempotency_key}={marker!r}",
        )

        # ================================================================
        # 5. Neo4j
        # ================================================================

        await neo4j.verify_connectivity()

        transaction_result = await neo4j.execute(
            """
            MATCH (t:Transaction {id: $transaction_id})
            RETURN
                t.id AS id,
                t.event_id AS event_id,
                t.event_time AS event_time,
                t.amount AS amount
            """,
            {
                "transaction_id": transaction_id
            },
        )

        check(
            "Neo4j Transaction node",
            bool(transaction_result),
            str(
                transaction_result[0]
                if transaction_result
                else None
            ),
        )

        relationship_result = await neo4j.execute(
            """
            MATCH
                (c:Customer {id: $customer_id})
                -[:MADE]->
                (t:Transaction {id: $transaction_id})

            OPTIONAL MATCH
                (t)-[:AT_MERCHANT]->(m:Merchant)

            MATCH
                (c)-[:USED_DEVICE]->
                (d:Device {id: $device_id})

            MATCH
                (c)-[:USED_IP]->
                (ip:IP {address: $ip_address})

            RETURN
                c.id AS customer_id,
                t.id AS transaction_id,
                m.id AS merchant_id,
                d.id AS device_id,
                ip.address AS ip_address
            """,
            {
                "customer_id": customer_id,
                "transaction_id": transaction_id,
                "device_id": device_id,
                "ip_address": ip_address,
            },
        )

        check(
            "Neo4j relationship projection",
            bool(relationship_result),
            str(
                relationship_result[0]
                if relationship_result
                else None
            ),
        )

        # ================================================================
        # Final result
        # ================================================================

        print("\n" + "=" * 88)
        print("R7 FAULT ISOLATION VERIFICATION: PASS")
        print("=" * 88)

        print(
            "PostgreSQL : transaction + decision + outbox verified"
        )
        print(
            "Redis      : velocity + idempotency verified"
        )
        print(
            "Neo4j      : transaction + graph relationships verified"
        )
        print(
            "Decision   : "
            f"{decision.decision} / {decision.risk_level}"
        )
        print(
            f"Probability: {actual_probability:.4f}"
        )
        print(
            f"Loss       : {actual_expected_loss:.2f} INR"
        )
        print(
            "Versions   : "
            f"{decision.model_version} / "
            f"{decision.calibration_version} / "
            f"{decision.policy_version}"
        )

    finally:
        await redis.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())