from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def rolling_count(
    timestamps: np.ndarray,
    keys: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    """
    Count previous transactions for the same key inside
    the preceding time window.

    The current transaction is excluded.
    """
    result = np.zeros(len(timestamps), dtype=np.int32)

    last_seen: dict[object, list[int]] = {}

    for i, (timestamp, key) in enumerate(zip(timestamps, keys)):
        history = last_seen.setdefault(key, [])

        cutoff = timestamp - window_seconds

        # Remove transactions outside the window.
        while history and timestamps[history[0]] < cutoff:
            history.pop(0)

        result[i] = len(history)

        history.append(i)

    return result


def rolling_unique_count(
    timestamps: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    """
    Count distinct values associated with each key inside
    the preceding time window.

    The current transaction is excluded.
    """
    result = np.zeros(len(timestamps), dtype=np.int32)

    histories: dict[object, list[tuple[int, object]]] = {}

    for i, (timestamp, key, value) in enumerate(
        zip(timestamps, keys, values)
    ):
        history = histories.setdefault(key, [])

        cutoff = timestamp - window_seconds

        while history and history[0][0] < cutoff:
            history.pop(0)

        result[i] = len({item[1] for item in history})

        history.append((timestamp, value))

    return result


def main() -> None:
    input_path = ARTIFACT_DIR / "train_combined.parquet"
    output_path = ARTIFACT_DIR / "train_online_features_v2.parquet"

    print("Loading dataset...")

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

    df = pd.read_parquet(
        input_path,
        columns=columns,
    )

    print(f"Rows loaded: {len(df):,}")

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------
    df = (
        df.sort_values(
            ["TransactionDT", "TransactionID"]
        )
        .reset_index(drop=True)
    )

    timestamps = df["TransactionDT"].to_numpy(
        dtype=np.int64
    )

    # ---------------------------------------------------------
    # Historical identity proxies
    # ---------------------------------------------------------
    customer = df["card1"].fillna(-1).to_numpy()

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
        + df["card6"]
        .fillna("__MISSING__")
        .astype(str)
    ).to_numpy()

    merchant = (
        df["ProductCD"]
        .fillna("__MISSING__")
        .astype(str)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Redis-equivalent velocity features
    # ---------------------------------------------------------
    print("Building customer 1-minute velocity...")

    customer_1m = rolling_count(
        timestamps,
        customer,
        60,
    )

    print("Building customer 1-hour velocity...")

    customer_1h = rolling_count(
        timestamps,
        customer,
        3600,
    )

    print("Building device 1-hour velocity...")

    device_1h = rolling_count(
        timestamps,
        device,
        3600,
    )

    print("Building IP 1-hour velocity...")

    ip_1h = rolling_count(
        timestamps,
        ip,
        3600,
    )

    # ---------------------------------------------------------
    # Historical relationship features
    # ---------------------------------------------------------
    print("Building payment 1-hour velocity...")

    payment_1h = rolling_count(
        timestamps,
        payment,
        3600,
    )

    print("Building merchant 1-hour velocity...")

    merchant_1h = rolling_count(
        timestamps,
        merchant,
        3600,
    )

    print("Building device/customer relationships...")

    device_customer = rolling_unique_count(
        timestamps,
        device,
        customer,
        3600,
    )

    print("Building IP/customer relationships...")

    ip_customer = rolling_unique_count(
        timestamps,
        ip,
        customer,
        3600,
    )

    print("Building payment/customer relationships...")

    payment_customer = rolling_unique_count(
        timestamps,
        payment,
        customer,
        3600,
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    result = pd.DataFrame(
        {
            "TransactionID": df["TransactionID"],
            "TransactionDT": df["TransactionDT"],
            "isFraud": df["isFraud"].astype("int8"),

            "amount": df["TransactionAmt"],
            "amount_log": np.log1p(
                df["TransactionAmt"].clip(lower=0)
            ),

            "customer_transactions_1m": customer_1m,
            "customer_transactions_1h": customer_1h,
            "device_transactions_1h": device_1h,
            "ip_transactions_1h": ip_1h,

            "payment_transactions_1h": payment_1h,
            "merchant_transactions_1h": merchant_1h,

            "device_customer_count": device_customer,
            "ip_customer_count": ip_customer,
            "payment_customer_count": payment_customer,

            "transaction_day": (
                df["TransactionDT"] // 86400
            ),

            "transaction_hour": (
                (df["TransactionDT"] % 86400) // 3600
            ),

            "transaction_week": (
                df["TransactionDT"] // (7 * 86400)
            ),

            "missing_count": df.isna().sum(axis=1),
        }
    )

    print("=" * 80)
    print("ONLINE-COMPATIBLE FEATURES")
    print("=" * 80)

    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns):,}")

    print("\nFeature columns:")

    for column in result.columns:
        print(f"  {column}")

    print("\nFraud distribution:")
    print(result["isFraud"].value_counts())

    print("\nVelocity statistics:")

    velocity_columns = [
        "customer_transactions_1m",
        "customer_transactions_1h",
        "device_transactions_1h",
        "ip_transactions_1h",
    ]

    print(
        result[velocity_columns]
        .describe()
        .to_string()
    )

    print("\nSaving...")

    result.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
