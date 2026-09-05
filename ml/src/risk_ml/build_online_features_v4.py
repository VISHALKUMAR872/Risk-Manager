from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

INPUT_PATH = ARTIFACT_DIR / "train_combined.parquet"
OUTPUT_PATH = ARTIFACT_DIR / "train_online_features_v4.parquet"

WINDOW_1M = 60
WINDOW_1H = 3600


FEATURE_COLUMNS = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
    "payment_customer_count",
    "merchant_transaction_count",
]


def main() -> None:
    print("=" * 80)
    print("RISK SENTINEL — ONLINE FEATURE REPLAY V4")
    print("=" * 80)

    columns = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "TransactionAmt",
        "card1",
        "DeviceInfo",
        "addr1",
        "addr2",
        "card5",
        "card6",
        "ProductCD",
    ]

    print("\nLoading dataset...")
    df = pd.read_parquet(INPUT_PATH, columns=columns)

    print(f"Rows loaded: {len(df):,}")

    # ------------------------------------------------------------------
    # Deterministic chronological ordering.
    #
    # TransactionID is only a deterministic tie-breaker.
    # IMPORTANT: transactions with the same TransactionDT are processed
    # together so they cannot see each other as historical state.
    # ------------------------------------------------------------------

    df = (
        df.sort_values(
            ["TransactionDT", "TransactionID"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    timestamps = df["TransactionDT"].to_numpy(dtype=np.int64)

    # ------------------------------------------------------------------
    # Dataset identity proxies.
    #
    # These are IEEE-CIS proxies, not claims about real production IDs.
    # ------------------------------------------------------------------

    customer = (
        df["card1"]
        .fillna(-1)
        .astype(np.int64)
        .to_numpy()
    )

    device = (
        df["DeviceInfo"]
        .fillna("__MISSING_DEVICE__")
        .astype(str)
        .to_numpy()
    )

    ip = (
        df["addr1"].fillna(-1).astype(str)
        + "_"
        + df["addr2"].fillna(-1).astype(str)
    ).to_numpy()

    payment = (
        df["card1"].fillna(-1).astype(str)
        + "_"
        + df["card5"].fillna(-1).astype(str)
        + "_"
        + df["card6"].fillna("__MISSING__").astype(str)
    ).to_numpy()

    merchant = (
        df["ProductCD"]
        .fillna("__MISSING_MERCHANT__")
        .astype(str)
        .to_numpy()
    )

    amount = (
        df["TransactionAmt"]
        .astype(float)
        .to_numpy()
    )

    target = (
        df["isFraud"]
        .astype("int8")
        .to_numpy()
    )

    n = len(df)

    # ------------------------------------------------------------------
    # Redis-equivalent rolling state.
    #
    # Each deque contains timestamps of historical transactions.
    #
    # State is queried BEFORE inserting the current timestamp.
    # ------------------------------------------------------------------

    customer_events_1m: dict[object, deque[int]] = defaultdict(deque)
    customer_events_1h: dict[object, deque[int]] = defaultdict(deque)
    device_events_1h: dict[object, deque[int]] = defaultdict(deque)
    ip_events_1h: dict[object, deque[int]] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Neo4j-equivalent historical graph state.
    #
    # customer -> number of MADE -> Transaction relationships
    #
    # and historical distinct associations:
    #
    # customer -> Device
    # customer -> IP
    # customer -> PaymentMethod
    #
    # ------------------------------------------------------------------

    customer_transaction_count: dict[object, int] = defaultdict(int)

    device_customers: dict[object, set] = defaultdict(set)
    ip_customers: dict[object, set] = defaultdict(set)
    payment_customers: dict[object, set] = defaultdict(set)

    # Historical merchant transaction count.
    merchant_transaction_count: dict[object, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Output arrays.
    # ------------------------------------------------------------------

    customer_1m = np.zeros(n, dtype=np.int32)
    customer_1h = np.zeros(n, dtype=np.int32)
    device_1h = np.zeros(n, dtype=np.int32)
    ip_1h = np.zeros(n, dtype=np.int32)

    customer_degree = np.zeros(n, dtype=np.int32)
    device_customer_count = np.zeros(n, dtype=np.int32)
    ip_customer_count = np.zeros(n, dtype=np.int32)
    payment_customer_count = np.zeros(n, dtype=np.int32)
    merchant_count = np.zeros(n, dtype=np.int32)

    # ------------------------------------------------------------------
    # Replay.
    #
    # We process one timestamp group at a time.
    #
    # Phase A:
    #   calculate features from state strictly before timestamp T.
    #
    # Phase B:
    #   update historical state with every transaction at T.
    #
    # This guarantees:
    #
    #       historical_event_time < scoring_event_time
    #
    # even when multiple events have identical timestamps.
    # ------------------------------------------------------------------

    print("\nReplaying historical state...")

    i = 0

    while i < n:
        timestamp = timestamps[i]

        j = i + 1

        while j < n and timestamps[j] == timestamp:
            j += 1

        # ==============================================================
        # PHASE A — SCORE CURRENT TIMESTAMP AGAINST PRIOR STATE
        # ==============================================================

        for k in range(i, j):
            c = customer[k]
            d = device[k]
            ip_key = ip[k]
            p = payment[k]
            m = merchant[k]

            cutoff_1m = timestamp - WINDOW_1M
            cutoff_1h = timestamp - WINDOW_1H

            # ----------------------------------------------------------
            # Redis: customer 1 minute
            # ----------------------------------------------------------

            customer_queue = customer_events_1m[c]

            while (
                customer_queue
                and customer_queue[0] <= cutoff_1m
            ):
                customer_queue.popleft()

            customer_1m[k] = len(customer_queue)

            # ----------------------------------------------------------
            # Redis: customer 1 hour
            # ----------------------------------------------------------

            customer_queue_1h = customer_events_1h[c]

            while (
                customer_queue_1h
                and customer_queue_1h[0] <= cutoff_1h
            ):
                customer_queue_1h.popleft()

            customer_1h[k] = len(customer_queue_1h)

            # ----------------------------------------------------------
            # Redis: device 1 hour
            # ----------------------------------------------------------

            device_queue = device_events_1h[d]

            while (
                device_queue
                and device_queue[0] <= cutoff_1h
            ):
                device_queue.popleft()

            device_1h[k] = len(device_queue)

            # ----------------------------------------------------------
            # Redis: IP 1 hour
            # ----------------------------------------------------------

            ip_queue = ip_events_1h[ip_key]

            while (
                ip_queue
                and ip_queue[0] <= cutoff_1h
            ):
                ip_queue.popleft()

            ip_1h[k] = len(ip_queue)

            # ----------------------------------------------------------
            # Neo4j: customer degree
            #
            # Exact conceptual equivalent:
            #
            #   number of historical MADE relationships
            #   +
            #   number of historical USED_DEVICE relationships
            #   +
            #   number of historical USED_IP relationships
            #   +
            #   number of historical USED_PAYMENT relationships
            # ----------------------------------------------------------

            device_customer_count[k] = len(device_customers[d])

            ip_customer_count[k] = len(
                ip_customers[ip_key]
            )

            payment_customer_count[k] = len(
                payment_customers[p]
            )

            merchant_count[k] = merchant_transaction_count[m]

            # The graph degree must use relationships belonging to
            # THIS CUSTOMER, not global entity counts.
            #
            # Maintain those relationship sets explicitly below.
            #
            # They are populated lazily here through separate state.

            # ----------------------------------------------------------
            # Historical distinct customers associated with entities.
            # ----------------------------------------------------------

            device_customer_count[k] = len(
                device_customers[d]
            )

            ip_customer_count[k] = len(
                ip_customers[ip_key]
            )

            payment_customer_count[k] = len(
                payment_customers[p]
            )

            # ----------------------------------------------------------
            # Merchant: all historical transactions, not 1-hour window.
            # ----------------------------------------------------------

            merchant_count[k] = merchant_transaction_count[m]

        # ==============================================================
        # PHASE B — UPDATE HISTORICAL STATE
        # ==============================================================

        for k in range(i, j):
            c = customer[k]
            d = device[k]
            ip_key = ip[k]
            p = payment[k]
            m = merchant[k]

            # Redis rolling state.
            customer_events_1m[c].append(timestamp)
            customer_events_1h[c].append(timestamp)
            device_events_1h[d].append(timestamp)
            ip_events_1h[ip_key].append(timestamp)

            # Neo4j historical relationships.
            customer_transaction_count[c] += 1

            device_customers[d].add(c)
            ip_customers[ip_key].add(c)
            payment_customers[p].add(c)

            merchant_transaction_count[m] += 1

        i = j

        if i % 50_000 == 0 or i == n:
            print(f"  Processed {i:,} / {n:,}")

    # ------------------------------------------------------------------
    # Correct customer degree.
    #
    # The live Neo4j implementation counts relationships attached to
    # the customer:
    #
    #   MADE
    #   USED_DEVICE
    #   USED_IP
    #   USED_PAYMENT
    #
    # Therefore we need customer-specific relationship cardinalities.
    #
    # Replaying those separately avoids confusing entity-global counts
    # with customer-local degree.
    # ------------------------------------------------------------------

    print("\nReplaying customer-specific graph degree...")

    customer_tx_count: dict[object, int] = defaultdict(int)
    customer_devices: dict[object, set] = defaultdict(set)
    customer_ips: dict[object, set] = defaultdict(set)
    customer_payments: dict[object, set] = defaultdict(set)

    i = 0

    while i < n:
        timestamp = timestamps[i]

        j = i + 1

        while j < n and timestamps[j] == timestamp:
            j += 1

        # Features before current timestamp.
        for k in range(i, j):
            c = customer[k]

            customer_degree[k] = (
                customer_tx_count[c]
                + len(customer_devices[c])
                + len(customer_ips[c])
                + len(customer_payments[c])
            )

        # Add current timestamp relationships.
        for k in range(i, j):
            c = customer[k]

            customer_tx_count[c] += 1
            customer_devices[c].add(device[k])
            customer_ips[c].add(ip[k])
            customer_payments[c].add(payment[k])

        i = j

    # ------------------------------------------------------------------
    # Construct final artifact.
    # ------------------------------------------------------------------

    result = pd.DataFrame(
        {
            "TransactionID": df["TransactionID"],
            "TransactionDT": df["TransactionDT"],
            "isFraud": target,
            "amount": amount,

            "customer_transactions_1m": customer_1m,
            "customer_transactions_1h": customer_1h,
            "device_transactions_1h": device_1h,
            "ip_transactions_1h": ip_1h,

            "customer_degree": customer_degree,
            "device_customer_count": device_customer_count,
            "ip_customer_count": ip_customer_count,
            "payment_customer_count": payment_customer_count,
            "merchant_transaction_count": merchant_count,
        }
    )

    # ------------------------------------------------------------------
    # Integrity checks.
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("V4 FEATURE ARTIFACT")
    print("=" * 80)

    print(f"Rows:    {len(result):,}")
    print(f"Columns: {len(result.columns):,}")

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in result.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing production features: {missing}"
        )

    if result[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError(
            "Feature artifact contains NaN values."
        )

    # Current transaction must never contribute to its own velocity.
    if (result["customer_transactions_1m"] < 0).any():
        raise RuntimeError("Negative 1m velocity detected.")

    if (result["customer_transactions_1h"] < 0).any():
        raise RuntimeError("Negative 1h velocity detected.")

    print("\nFraud distribution:")
    print(result["isFraud"].value_counts())

    print("\nFeature statistics:")
    print(
        result[FEATURE_COLUMNS]
        .describe()
        .to_string()
    )

    print("\nSaving...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nV4 replay complete.")


if __name__ == "__main__":
    main()
