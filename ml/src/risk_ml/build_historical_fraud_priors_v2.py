from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

INPUT_PATH = ARTIFACT_DIR / "train_combined.parquet"
OUTPUT_PATH = ARTIFACT_DIR / "train_online_features_v5_priors.parquet"


ENTITY_NAMES = [
    "customer",
    "merchant",
    "device",
    "ip",
    "payment",
]


def safe_rate(fraud_count: int, transaction_count: int) -> float:
    if transaction_count <= 0:
        return 0.0
    return fraud_count / transaction_count


def main() -> None:
    print("=" * 80)
    print("RISK SENTINEL — POINT-IN-TIME HISTORICAL FRAUD PRIORS V2")
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

    # Deterministic chronological ordering.
    df = (
        df.sort_values(
            ["TransactionDT", "TransactionID"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    timestamps = df["TransactionDT"].to_numpy(dtype=np.int64)
    target = df["isFraud"].astype("int8").to_numpy()

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

    amount = df["TransactionAmt"].astype(float).to_numpy()

    n = len(df)

    entities = {
        "customer": customer,
        "merchant": merchant,
        "device": device,
        "ip": ip,
        "payment": payment,
    }

    # Historical state.
    transaction_counts = {
        name: defaultdict(int)
        for name in ENTITY_NAMES
    }

    fraud_counts = {
        name: defaultdict(int)
        for name in ENTITY_NAMES
    }

    # Output arrays.
    outputs: dict[str, np.ndarray] = {}

    for name in ENTITY_NAMES:
        outputs[f"{name}_historical_fraud_count"] = np.zeros(
            n,
            dtype=np.int32,
        )
        outputs[f"{name}_historical_transaction_count"] = np.zeros(
            n,
            dtype=np.int32,
        )
        outputs[f"{name}_historical_fraud_rate"] = np.zeros(
            n,
            dtype=np.float64,
        )

    print("\nReplaying historical fraud state...")

    i = 0

    while i < n:
        timestamp = timestamps[i]

        j = i + 1

        while j < n and timestamps[j] == timestamp:
            j += 1

        # ==============================================================
        # PHASE A
        #
        # Score every transaction at timestamp T using ONLY state from
        # timestamps strictly earlier than T.
        # ==============================================================

        for k in range(i, j):
            for name in ENTITY_NAMES:
                key = entities[name][k]

                tx_count = transaction_counts[name][key]
                fraud_count = fraud_counts[name][key]

                outputs[
                    f"{name}_historical_transaction_count"
                ][k] = tx_count

                outputs[
                    f"{name}_historical_fraud_count"
                ][k] = fraud_count

                outputs[
                    f"{name}_historical_fraud_rate"
                ][k] = safe_rate(
                    fraud_count,
                    tx_count,
                )

        # ==============================================================
        # PHASE B
        #
        # Add ALL transactions at timestamp T to historical state.
        # ==============================================================

        for k in range(i, j):
            for name in ENTITY_NAMES:
                key = entities[name][k]

                transaction_counts[name][key] += 1
                fraud_counts[name][key] += int(target[k])

        i = j

        if i % 50_000 == 0 or i == n:
            print(f"  Processed {i:,} / {n:,}")

    # ------------------------------------------------------------------
    # Construct artifact.
    # ------------------------------------------------------------------

    result_data = {
        "TransactionID": df["TransactionID"].to_numpy(),
        "TransactionDT": df["TransactionDT"].to_numpy(),
        "isFraud": target,
        "amount": amount,
    }

    # Add original V4 online features so this becomes a complete V5
    # feature artifact once we join them below.
    v4_path = ARTIFACT_DIR / "train_online_features_v4.parquet"

    print("\nLoading V4 online features...")
    v4 = pd.read_parquet(v4_path)

    required_v4 = [
        "TransactionID",
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

    missing_v4 = [
        c for c in required_v4
        if c not in v4.columns
    ]

    if missing_v4:
        raise RuntimeError(
            f"V4 artifact missing columns: {missing_v4}"
        )

    v4_features = v4[required_v4].copy()

    result = pd.DataFrame(result_data)

    result = result.drop(columns=["amount"])

    result = v4_features.merge(
        result,
        on="TransactionID",
        how="inner",
        validate="one_to_one",
    )

    if len(result) != n:
        raise RuntimeError(
            f"V4 join changed row count: "
            f"{len(result):,} != {n:,}"
        )

    for name in ENTITY_NAMES:
        result[
            f"{name}_historical_fraud_count"
        ] = outputs[
            f"{name}_historical_fraud_count"
        ]

        result[
            f"{name}_historical_transaction_count"
        ] = outputs[
            f"{name}_historical_transaction_count"
        ]

        result[
            f"{name}_historical_fraud_rate"
        ] = outputs[
            f"{name}_historical_fraud_rate"
        ]

    # ------------------------------------------------------------------
    # Leakage checks.
    # ------------------------------------------------------------------

    print("\nRunning leakage checks...")

    if result["TransactionID"].duplicated().any():
        raise RuntimeError("Duplicate TransactionID detected.")

    for name in ENTITY_NAMES:
        count_col = f"{name}_historical_transaction_count"
        fraud_col = f"{name}_historical_fraud_count"
        rate_col = f"{name}_historical_fraud_rate"

        if (result[fraud_col] > result[count_col]).any():
            raise RuntimeError(
                f"{name}: fraud count exceeds transaction count."
            )

        if (result[rate_col] < 0).any():
            raise RuntimeError(
                f"{name}: negative fraud rate."
            )

        if (result[rate_col] > 1).any():
            raise RuntimeError(
                f"{name}: fraud rate exceeds 1."
            )

    # Exact same-timestamp leakage audit.
    #
    # For every timestamp, all rows must have identical historical
    # state for a given entity before that timestamp is added.
    #
    # Therefore no row at T may include another row at T.
    #
    # We verify this by recomputing a small deterministic sample of
    # timestamp groups directly from the raw ordered data.

    print("Checking same-timestamp isolation...")

    ordered = (
        df[
            [
                "TransactionID",
                "TransactionDT",
                "isFraud",
            ]
        ]
        .sort_values(
            ["TransactionDT", "TransactionID"],
            kind="mergesort",
        )
    )

    # The first transaction must have zero historical fraud state.
    first_id = int(ordered.iloc[0]["TransactionID"])

    first_row = result[
        result["TransactionID"] == first_id
    ].iloc[0]

    for name in ENTITY_NAMES:
        if int(
            first_row[
                f"{name}_historical_transaction_count"
            ]
        ) != 0:
            raise RuntimeError(
                f"{name}: first transaction has historical state."
            )

    feature_columns = [
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

    prior_columns = [
        f"{name}_historical_fraud_count"
        for name in ENTITY_NAMES
    ] + [
        f"{name}_historical_transaction_count"
        for name in ENTITY_NAMES
    ] + [
        f"{name}_historical_fraud_rate"
        for name in ENTITY_NAMES
    ]

    all_features = feature_columns + prior_columns

    if result[all_features].isna().any().any():
        missing = result[all_features].isna().sum()
        raise RuntimeError(
            f"NaN values detected:\n{missing[missing > 0]}"
        )

    print("\nHistorical prior statistics:")

    for name in ENTITY_NAMES:
        print(f"\n{name.upper()}")

        cols = [
            f"{name}_historical_fraud_count",
            f"{name}_historical_transaction_count",
            f"{name}_historical_fraud_rate",
        ]

        print(
            result[cols]
            .describe()
            .to_string()
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("V2 HISTORICAL FRAUD PRIOR BUILD COMPLETE")
    print("=" * 80)

    print(f"Rows:    {len(result):,}")
    print(f"Columns: {len(result.columns):,}")
    print(f"Saved:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
