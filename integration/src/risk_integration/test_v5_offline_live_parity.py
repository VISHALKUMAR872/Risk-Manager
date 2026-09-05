from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from risk_engine.contracts import TransactionEvent
from risk_features.online import OnlineFeatureService
from risk_persistence.neo4j import Neo4jClient
from risk_persistence.redis import RedisClient


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS = ROOT / "ml" / "artifacts"

RAW_DATASET = ARTIFACTS / "train_combined.parquet"
OFFLINE_FEATURES = ARTIFACTS / "train_online_features_v4.parquet"


# ============================================================================
# V5 production feature contract
# ============================================================================

V5_FEATURE_COLUMNS = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]

RAW_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "DeviceInfo",
    "addr1",
    "addr2",
    "card5",
    "card6",
    "ProductCD",
]

CHECKPOINT_COUNT = 10

# Keep Neo4j requests reasonably sized.
NEO4J_BATCH_SIZE = 1000

# Redis pipeline batch size.
REDIS_BATCH_SIZE = 1000

NAMESPACE = "v5-parity"


# ============================================================================
# IEEE-CIS identity proxies
# ============================================================================


def proxy_customer(value) -> str:
    if pd.isna(value):
        return "__MISSING_CUSTOMER__"

    return f"ieee-customer-{value}"


def proxy_device(value) -> str:
    if pd.isna(value):
        return "__MISSING_DEVICE__"

    return str(value)


def proxy_ip(addr1, addr2) -> str:
    left = "-1" if pd.isna(addr1) else str(addr1)
    right = "-1" if pd.isna(addr2) else str(addr2)

    return f"{left}_{right}"


def proxy_payment(card1, card5, card6) -> str:
    c1 = "-1" if pd.isna(card1) else str(card1)
    c5 = "-1" if pd.isna(card5) else str(card5)
    c6 = "__MISSING__" if pd.isna(card6) else str(card6)

    return f"{c1}_{c5}_{c6}"


def proxy_merchant(value) -> str:
    if pd.isna(value):
        return "__MISSING_MERCHANT__"

    return str(value)


# ============================================================================
# TransactionDT -> datetime
# ============================================================================


def transaction_datetime(transaction_dt: int) -> datetime:
    return (
        datetime(
            1970,
            1,
            1,
            tzinfo=timezone.utc,
        )
        + timedelta(seconds=int(transaction_dt))
    )


# ============================================================================
# TransactionEvent conversion
# ============================================================================


def row_to_event(row: pd.Series) -> TransactionEvent:
    transaction_id = str(int(row["TransactionID"]))

    return TransactionEvent(
        event_id=f"{NAMESPACE}-event-{transaction_id}",
        transaction_id=f"{NAMESPACE}-txn-{transaction_id}",
        event_time=transaction_datetime(
            int(row["TransactionDT"])
        ),
        customer_id=(
            f"{NAMESPACE}-customer-"
            f"{proxy_customer(row['card1'])}"
        ),
        merchant_id=(
            f"{NAMESPACE}-merchant-"
            f"{proxy_merchant(row['ProductCD'])}"
        ),
        amount=float(row["TransactionAmt"]),
        currency="INR",
        device_id=(
            f"{NAMESPACE}-device-"
            f"{proxy_device(row['DeviceInfo'])}"
        ),
        ip_address=(
            f"{NAMESPACE}-ip-"
            f"{proxy_ip(row['addr1'], row['addr2'])}"
        ),
        payment_method_id=(
            f"{NAMESPACE}-payment-"
            f"{proxy_payment(row['card1'], row['card5'], row['card6'])}"
        ),
        merchant_category="ieee-cis",
        country="XX",
        channel="offline-parity",
    )


# ============================================================================
# Assertions
# ============================================================================


def compare_feature(
    *,
    transaction_id: int,
    feature: str,
    offline_value,
    live_value,
) -> None:
    if feature == "amount":
        left = float(offline_value)
        right = float(live_value)

        if abs(left - right) > 1e-9:
            raise AssertionError(
                f"TransactionID={transaction_id} "
                f"{feature}: offline={left}, live={right}"
            )

        return

    left = int(offline_value)
    right = int(live_value)

    if left != right:
        raise AssertionError(
            f"TransactionID={transaction_id} "
            f"{feature}: offline={left}, live={right}"
        )


# ============================================================================
# Checkpoint selection
# ============================================================================


def select_checkpoints(
    df: pd.DataFrame,
) -> pd.DataFrame:
    candidates = df[
        df["card1"].notna()
        & df["DeviceInfo"].notna()
        & df["addr1"].notna()
        & df["addr2"].notna()
    ].copy()

    if len(candidates) < CHECKPOINT_COUNT:
        raise RuntimeError(
            "Not enough valid checkpoint candidates."
        )

    positions = [
        int(
            round(
                i * (len(candidates) - 1)
                / (CHECKPOINT_COUNT - 1)
            )
        )
        for i in range(CHECKPOINT_COUNT)
    ]

    return (
        candidates.iloc[positions]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================================
# Dependency closure
# ============================================================================


def build_dependency_union(
    df: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the union of all historical rows required by the selected
    checkpoints.

    IMPORTANT:
    Historical state is strictly prior to the checkpoint TransactionDT.

    We deliberately do NOT include same-timestamp rows, even when their
    TransactionID is smaller. This matches the offline same-timestamp group
    isolation contract and the live providers' strict '< event_time'
    semantics.
    """

    historical_ids: set[int] = set()

    for checkpoint_index, target in checkpoints.iterrows():
        target_dt = int(target["TransactionDT"])

        historical = df[
            df["TransactionDT"] < target_dt
        ]

        if historical.empty:
            print(
                f"  checkpoint {checkpoint_index + 1:02d}/"
                f"{len(checkpoints)}: "
                f"TransactionID={int(target['TransactionID']):,} "
                f"dependencies=0"
            )
            continue

        customer = target["card1"]
        device = target["DeviceInfo"]
        addr1 = target["addr1"]
        addr2 = target["addr2"]

        if pd.isna(customer):
            customer_mask = historical["card1"].isna()
        else:
            customer_mask = historical["card1"].eq(customer)

        if pd.isna(device):
            device_mask = historical["DeviceInfo"].isna()
        else:
            device_mask = historical["DeviceInfo"].eq(device)

        if pd.isna(addr1):
            ip_addr1_mask = historical["addr1"].isna()
        else:
            ip_addr1_mask = historical["addr1"].eq(addr1)

        if pd.isna(addr2):
            ip_addr2_mask = historical["addr2"].isna()
        else:
            ip_addr2_mask = historical["addr2"].eq(addr2)

        ip_mask = ip_addr1_mask & ip_addr2_mask

        closure = historical.loc[
            customer_mask
            | device_mask
            | ip_mask
        ]

        ids = closure["TransactionID"].astype(int).tolist()

        historical_ids.update(ids)

        print(
            f"  checkpoint {checkpoint_index + 1:02d}/"
            f"{len(checkpoints)}: "
            f"TransactionID={int(target['TransactionID']):,} "
            f"dependencies={len(ids):,}"
        )

    if not historical_ids:
        return df.iloc[0:0].copy()

    dependency_df = (
        df[
            df["TransactionID"].isin(historical_ids)
        ]
        .sort_values(
            ["TransactionDT", "TransactionID"]
        )
        .reset_index(drop=True)
    )

    return dependency_df


# ============================================================================
# Redis batched replay
# ============================================================================


async def replay_redis_batch(
    redis: RedisClient,
    events: list[TransactionEvent],
) -> None:
    """
    Reproduce VelocityStore.record_transaction() using Redis pipelines.

    This is deliberately equivalent to the production data structures:

      customer:{id}:tx_events
      device:{id}:tx_events
      ip:{id}:tx_events
      velocity:processed:{transaction_id}

    The idempotency marker uses NX semantics.
    """

    if not events:
        return

    pipe = redis.client.pipeline(transaction=False)

    for event in events:
        timestamp = event.event_time.timestamp()

        idempotency_key = (
            f"velocity:processed:{event.transaction_id}"
        )

        customer_key = (
            f"customer:{event.customer_id}:tx_events"
        )

        device_key = (
            f"device:{event.device_id}:tx_events"
        )

        ip_key = (
            f"ip:{event.ip_address}:tx_events"
        )

        # We need idempotency semantics equivalent to the production
        # record_velocity_event() Lua operation.
        #
        # Because this replay set contains unique transaction IDs, NX is
        # sufficient here and preserves the intended final Redis state.
        pipe.set(
            idempotency_key,
            "1",
            ex=3600,
            nx=True,
        )

        pipe.zadd(
            customer_key,
            {event.transaction_id: timestamp},
        )

        pipe.zadd(
            device_key,
            {event.transaction_id: timestamp},
        )

        pipe.zadd(
            ip_key,
            {event.transaction_id: timestamp},
        )

    await pipe.execute()


async def replay_redis(
    redis: RedisClient,
    events: list[TransactionEvent],
) -> None:
    for start in range(
        0,
        len(events),
        REDIS_BATCH_SIZE,
    ):
        batch = events[
            start:start + REDIS_BATCH_SIZE
        ]

        await replay_redis_batch(
            redis,
            batch,
        )


# ============================================================================
# Neo4j batched replay
# ============================================================================


NEO4J_BATCH_QUERY = """
UNWIND $events AS e

MERGE (c:Customer {id: e.customer_id})
MERGE (m:Merchant {id: e.merchant_id})
MERGE (d:Device {id: e.device_id})
MERGE (ip:IP {address: e.ip_address})
MERGE (p:PaymentMethod {id: e.payment_method_id})

MERGE (t:Transaction {id: e.transaction_id})
SET
    t.event_id = e.event_id,
    t.amount = e.amount,
    t.currency = e.currency,
    t.event_time = datetime(e.event_time),
    t.merchant_category = e.merchant_category,
    t.country = e.country,
    t.channel = e.channel

MERGE (c)-[:MADE]->(t)

MERGE (t)-[:AT_MERCHANT]->(m)

MERGE (c)-[used_device:USED_DEVICE]->(d)
ON CREATE SET used_device.first_seen_at = datetime(e.event_time)
ON MATCH SET used_device.first_seen_at =
    CASE
        WHEN used_device.first_seen_at > datetime(e.event_time)
        THEN datetime(e.event_time)
        ELSE used_device.first_seen_at
    END

MERGE (c)-[used_ip:USED_IP]->(ip)
ON CREATE SET used_ip.first_seen_at = datetime(e.event_time)
ON MATCH SET used_ip.first_seen_at =
    CASE
        WHEN used_ip.first_seen_at > datetime(e.event_time)
        THEN datetime(e.event_time)
        ELSE used_ip.first_seen_at
    END

MERGE (c)-[used_payment:USED_PAYMENT]->(p)
ON CREATE SET used_payment.first_seen_at = datetime(e.event_time)
ON MATCH SET used_payment.first_seen_at =
    CASE
        WHEN used_payment.first_seen_at > datetime(e.event_time)
        THEN datetime(e.event_time)
        ELSE used_payment.first_seen_at
    END

RETURN count(*) AS processed
"""


def event_to_neo4j_dict(
    event: TransactionEvent,
) -> dict:
    return {
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
    }


async def replay_neo4j_batch(
    neo4j: Neo4jClient,
    events: list[TransactionEvent],
) -> None:
    if not events:
        return

    await neo4j.execute(
        NEO4J_BATCH_QUERY,
        {
            "events": [
                event_to_neo4j_dict(event)
                for event in events
            ]
        },
    )


async def replay_neo4j(
    neo4j: Neo4jClient,
    events: list[TransactionEvent],
) -> None:
    for start in range(
        0,
        len(events),
        NEO4J_BATCH_SIZE,
    ):
        batch = events[
            start:start + NEO4J_BATCH_SIZE
        ]

        await replay_neo4j_batch(
            neo4j,
            batch,
        )

        completed = min(
            start + len(batch),
            len(events),
        )

        if completed % 10000 == 0 or completed == len(events):
            print(
                f"    Neo4j replayed "
                f"{completed:,}/{len(events):,}"
            )


# ============================================================================
# Combined state replay
# ============================================================================


async def replay_state_batch(
    redis: RedisClient,
    neo4j: Neo4jClient,
    events: list[TransactionEvent],
) -> None:
    """
    Replay exactly the same historical event set into both online stores.

    Redis and Neo4j are independent state stores, so they can safely be
    populated concurrently for each batch.
    """

    await asyncio.gather(
        replay_redis(
            redis,
            events,
        ),
        replay_neo4j(
            neo4j,
            events,
        ),
    )


# ============================================================================
# Redis cleanup
# ============================================================================


async def cleanup_redis(
    redis: RedisClient,
) -> None:
    patterns = [
        f"customer:{NAMESPACE}-*",
        f"device:{NAMESPACE}-*",
        f"ip:{NAMESPACE}-*",
        f"velocity:processed:{NAMESPACE}-*",
    ]

    keys: set[str] = set()

    for pattern in patterns:
        async for key in redis.client.scan_iter(
            match=pattern,
            count=1000,
        ):
            keys.add(key)

    if keys:
        key_list = list(keys)

        for start in range(
            0,
            len(key_list),
            1000,
        ):
            await redis.client.delete(
                *key_list[start:start + 1000]
            )


# ============================================================================
# Neo4j cleanup
# ============================================================================


async def cleanup_neo4j(
    neo4j: Neo4jClient,
) -> None:
    queries = [
        """
        MATCH (t:Transaction)
        WHERE t.id STARTS WITH $prefix
        DETACH DELETE t
        """,
        """
        MATCH (c:Customer)
        WHERE c.id STARTS WITH $prefix
        DETACH DELETE c
        """,
        """
        MATCH (d:Device)
        WHERE d.id STARTS WITH $prefix
        DETACH DELETE d
        """,
        """
        MATCH (ip:IP)
        WHERE ip.address STARTS WITH $prefix
        DETACH DELETE ip
        """,
        """
        MATCH (p:PaymentMethod)
        WHERE p.id STARTS WITH $prefix
        DETACH DELETE p
        """,
        """
        MATCH (m:Merchant)
        WHERE m.id STARTS WITH $prefix
        DETACH DELETE m
        """,
    ]

    prefixes = [
        f"{NAMESPACE}-txn-",
        f"{NAMESPACE}-customer-",
        f"{NAMESPACE}-device-",
        f"{NAMESPACE}-ip-",
        f"{NAMESPACE}-payment-",
        f"{NAMESPACE}-merchant-",
    ]

    parameters = [
        {"prefix": prefix}
        for prefix in prefixes
    ]

    for query, params in zip(
        queries,
        parameters,
    ):
        await neo4j.execute(
            query,
            params,
        )


# ============================================================================
# Build dependency timeline
# ============================================================================


def build_timeline(
    dependency_df: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one chronological replay timeline.

    Every checkpoint is explicitly included, even if it has zero historical
    dependencies.

    Checkpoints are marked with _is_checkpoint=True.

    Same-timestamp rows are sorted by TransactionID, but the scoring rule is
    handled explicitly by the caller: all checkpoint features are queried
    before that checkpoint is projected.
    """

    dependency_ids = set(
        dependency_df["TransactionID"].astype(int)
    )

    checkpoint_ids = set(
        checkpoints["TransactionID"].astype(int)
    )

    timeline_ids = dependency_ids | checkpoint_ids

    timeline = (
        pd.concat(
            [
                dependency_df,
                checkpoints[
                    checkpoints["TransactionID"].isin(
                        checkpoint_ids
                    )
                ],
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["TransactionID"],
            keep="first",
        )
    )

    timeline = timeline[
        timeline["TransactionID"].isin(
            timeline_ids
        )
    ].copy()

    checkpoint_id_set = checkpoint_ids

    timeline["_is_checkpoint"] = (
        timeline["TransactionID"].astype(int).isin(
            checkpoint_id_set
        )
    )

    return (
        timeline
        .sort_values(
            ["TransactionDT", "TransactionID"]
        )
        .reset_index(drop=True)
    )


# ============================================================================
# Main parity gate
# ============================================================================


async def main() -> None:
    print("=" * 88)
    print(
        "RISK SENTINEL — V5 OFFLINE / LIVE FEATURE PARITY GATE"
    )
    print("=" * 88)

    # ------------------------------------------------------------------------
    # Validate artifacts
    # ------------------------------------------------------------------------

    if not RAW_DATASET.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {RAW_DATASET}"
        )

    if not OFFLINE_FEATURES.exists():
        raise FileNotFoundError(
            f"Missing offline feature artifact: "
            f"{OFFLINE_FEATURES}"
        )

    print("\nLoading offline artifacts...")

    df = pd.read_parquet(
        RAW_DATASET,
        columns=RAW_COLUMNS,
    )

    offline = pd.read_parquet(
        OFFLINE_FEATURES,
    )

    print(
        f"Raw transactions:     {len(df):,}"
    )
    print(
        f"Offline feature rows: {len(offline):,}"
    )

    # ------------------------------------------------------------------------
    # Validate columns
    # ------------------------------------------------------------------------

    required_offline = [
        "TransactionID",
        "TransactionDT",
        *V5_FEATURE_COLUMNS,
    ]

    missing_offline = [
        column
        for column in required_offline
        if column not in offline.columns
    ]

    if missing_offline:
        raise ValueError(
            "Offline artifact missing columns: "
            f"{missing_offline}"
        )

    missing_raw = [
        column
        for column in RAW_COLUMNS
        if column not in df.columns
    ]

    if missing_raw:
        raise ValueError(
            f"Raw dataset missing columns: {missing_raw}"
        )

    # ------------------------------------------------------------------------
    # Sort exactly like offline replay
    # ------------------------------------------------------------------------

    df = (
        df
        .sort_values(
            ["TransactionDT", "TransactionID"]
        )
        .reset_index(drop=True)
    )

    offline = (
        offline
        .sort_values(
            ["TransactionDT", "TransactionID"]
        )
        .reset_index(drop=True)
    )

    if len(df) != len(offline):
        raise AssertionError(
            "Raw and offline artifact row counts differ: "
            f"raw={len(df)}, offline={len(offline)}"
        )

    # ------------------------------------------------------------------------
    # Validate one-to-one TransactionID alignment
    # ------------------------------------------------------------------------

    if df["TransactionID"].duplicated().any():
        raise AssertionError(
            "Raw dataset contains duplicate TransactionID values."
        )

    if offline["TransactionID"].duplicated().any():
        raise AssertionError(
            "Offline feature artifact contains duplicate "
            "TransactionID values."
        )

    raw_ids = set(
        df["TransactionID"].astype(int)
    )

    offline_ids = set(
        offline["TransactionID"].astype(int)
    )

    if raw_ids != offline_ids:
        raise AssertionError(
            "Raw and offline TransactionID sets differ."
        )

    offline_by_id = offline.set_index(
        "TransactionID"
    )

    # ------------------------------------------------------------------------
    # Select checkpoints
    # ------------------------------------------------------------------------

    checkpoints = select_checkpoints(df)

    print(
        f"\nSelected checkpoints: "
        f"{len(checkpoints)}"
    )

    for index, row in checkpoints.iterrows():
        print(
            f"  {index + 1:02d}. "
            f"TransactionID="
            f"{int(row['TransactionID']):,} "
            f"TransactionDT="
            f"{int(row['TransactionDT']):,}"
        )

    # ------------------------------------------------------------------------
    # Build dependency closure
    # ------------------------------------------------------------------------

    print(
        "\nBuilding checkpoint dependency closure..."
    )

    dependency_df = build_dependency_union(
        df,
        checkpoints,
    )

    print(
        f"\nUnique historical dependency rows: "
        f"{len(dependency_df):,}"
    )

    # ------------------------------------------------------------------------
    # Build complete timeline
    # ------------------------------------------------------------------------

    timeline = build_timeline(
        dependency_df,
        checkpoints,
    )

    checkpoint_lookup = {
        int(row["TransactionID"]): row
        for _, row in checkpoints.iterrows()
    }

    print(
        f"Timeline rows including checkpoints: "
        f"{len(timeline):,}"
    )

    # ------------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------------

    redis = RedisClient()
    neo4j = Neo4jClient()

    service = OnlineFeatureService(
        redis=redis,
        neo4j=neo4j,
    )

    verified_checkpoints = 0
    replayed_ids: set[int] = set()

    try:
        await redis.ping()
        await neo4j.verify_connectivity()

        print(
            "\nCleaning previous parity state..."
        )

        await cleanup_redis(redis)
        await cleanup_neo4j(neo4j)

        # --------------------------------------------------------------------
        # Chronological replay
        # --------------------------------------------------------------------

        print(
            "\nRunning batched chronological "
            "offline/live replay..."
        )

        pending_events: list[TransactionEvent] = []

        for _, row in timeline.iterrows():
            transaction_id = int(
                row["TransactionID"]
            )

            is_checkpoint = bool(
                row["_is_checkpoint"]
            )

            # ---------------------------------------------------------------
            # Checkpoint scoring happens BEFORE projection.
            #
            # This is the critical point-in-time rule.
            # ---------------------------------------------------------------

            if is_checkpoint:
                # -----------------------------------------------------------
                # Flush all preceding historical state FIRST.
                # -----------------------------------------------------------

                if pending_events:
                    await replay_state_batch(
                        redis,
                        neo4j,
                        pending_events,
                    )
                    pending_events.clear()

                target = checkpoint_lookup[
                    transaction_id
                ]

                target_offline = offline_by_id.loc[
                    transaction_id
                ]

                target_event = row_to_event(
                    target
                )

                # Target is scored before it is projected.
                live_features = await service.build(
                    target_event
                )

                for feature in V5_FEATURE_COLUMNS:
                    compare_feature(
                        transaction_id=transaction_id,
                        feature=feature,
                        offline_value=target_offline[
                            feature
                        ],
                        live_value=getattr(
                            live_features,
                            feature,
                        ),
                    )

                verified_checkpoints += 1

                print(
                    f"[PASS] checkpoint "
                    f"{verified_checkpoints:02d}/"
                    f"{len(checkpoints):02d} "
                    f"TransactionID="
                    f"{transaction_id:,}"
                )


            # ---------------------------------------------------------------
            # Do not replay the same transaction twice.
            # ---------------------------------------------------------------

            if transaction_id in replayed_ids:
                continue

            event = row_to_event(row)

            pending_events.append(event)
            replayed_ids.add(transaction_id)

            # ---------------------------------------------------------------
            # Flush state periodically.
            #
            # NOTE:
            # A checkpoint is scored before this event is added to state.
            # Therefore the target cannot leak into its own features.
            # ---------------------------------------------------------------

            if len(pending_events) >= NEO4J_BATCH_SIZE:
                await replay_state_batch(
                    redis,
                    neo4j,
                    pending_events,
                )

                pending_events.clear()

                print(
                    f"    Historical state replayed: "
                    f"{len(replayed_ids):,}"
                )

        # --------------------------------------------------------------------
        # Final pending batch
        # --------------------------------------------------------------------

        if pending_events:
            await replay_state_batch(
                redis,
                neo4j,
                pending_events,
            )

        # --------------------------------------------------------------------
        # Final verification
        # --------------------------------------------------------------------

        if verified_checkpoints != len(checkpoints):
            raise AssertionError(
                "Checkpoint verification incomplete: "
                f"verified={verified_checkpoints}, "
                f"expected={len(checkpoints)}"
            )

        print("\n" + "=" * 88)
        print(
            "V5 OFFLINE / LIVE FEATURE PARITY: PASS"
        )
        print("=" * 88)

        print(
            f"Checkpoints verified: "
            f"{verified_checkpoints}"
        )

        print(
            f"V5 features verified: "
            f"{len(V5_FEATURE_COLUMNS)}"
        )

        print(
            f"Historical events replayed: "
            f"{len(replayed_ids):,}"
        )

        print("\nVerified feature contract:")

        for feature in V5_FEATURE_COLUMNS:
            print(
                f"  [PASS] {feature}"
            )

    finally:
        print(
            "\nCleaning parity state..."
        )

        try:
            await cleanup_redis(redis)
        except Exception as exc:
            print(
                f"[WARN] Redis cleanup failed: "
                f"{exc}"
            )

        try:
            await cleanup_neo4j(neo4j)
        except Exception as exc:
            print(
                f"[WARN] Neo4j cleanup failed: "
                f"{exc}"
            )

        await redis.close()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())