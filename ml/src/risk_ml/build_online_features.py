from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def build_online_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    result = pd.DataFrame(index=df.index)

    # ---------------------------------------------------------
    # Basic identifiers / target
    # ---------------------------------------------------------
    result["TransactionID"] = df["TransactionID"]
    result["TransactionDT"] = df["TransactionDT"]
    result["isFraud"] = df["isFraud"].astype("int8")

    # ---------------------------------------------------------
    # Transaction features
    # ---------------------------------------------------------
    result["amount"] = df["TransactionAmt"]

    result["amount_log"] = (
        df["TransactionAmt"].clip(lower=0).add(1).map(
            lambda x: __import__("math").log(x)
        )
    )

    # ---------------------------------------------------------
    # Historical identity proxies
    #
    # These are training-time proxies for the identifiers
    # available in the production online feature layer.
    # ---------------------------------------------------------
    customer_key = df["card1"]

    device_key = (
        df["DeviceInfo"]
        .fillna("__MISSING_DEVICE__")
        .astype(str)
    )

    ip_key = (
        df["addr1"].fillna(-1).astype(str)
        + "_"
        + df["addr2"].fillna(-1).astype(str)
    )

    payment_key = (
        df["card1"].astype(str)
        + "_"
        + df["card5"].fillna(-1).astype(str)
        + "_"
        + df["card6"].fillna("__MISSING__").astype(str)
    )

    merchant_key = df["ProductCD"].fillna("__MISSING__").astype(str)

    # ---------------------------------------------------------
    # Previous transaction counts
    #
    # IMPORTANT:
    # shift(1) ensures the current transaction itself is not
    # included in its own historical features.
    # ---------------------------------------------------------
    result["customer_transaction_count"] = (
        customer_key
        .groupby(customer_key)
        .cumcount()
    )

    result["device_transaction_count"] = (
        device_key
        .groupby(device_key)
        .cumcount()
    )

    result["ip_transaction_count"] = (
        ip_key
        .groupby(ip_key)
        .cumcount()
    )

    result["payment_transaction_count"] = (
        payment_key
        .groupby(payment_key)
        .cumcount()
    )

    result["merchant_transaction_count"] = (
        merchant_key
        .groupby(merchant_key)
        .cumcount()
    )

    # ---------------------------------------------------------
    # Previous customer/device relationships
    # ---------------------------------------------------------
    pair_customer_device = (
        customer_key.astype(str)
        + "_"
        + device_key
    )

    pair_customer_ip = (
        customer_key.astype(str)
        + "_"
        + ip_key
    )

    pair_customer_payment = (
        customer_key.astype(str)
        + "_"
        + payment_key
    )

    result["customer_device_transaction_count"] = (
        pair_customer_device
        .groupby(pair_customer_device)
        .cumcount()
    )

    result["customer_ip_transaction_count"] = (
        pair_customer_ip
        .groupby(pair_customer_ip)
        .cumcount()
    )

    result["customer_payment_transaction_count"] = (
        pair_customer_payment
        .groupby(pair_customer_payment)
        .cumcount()
    )

    # ---------------------------------------------------------
    # Number of distinct customers previously associated with
    # device / IP / payment key.
    #
    # We calculate these incrementally to avoid future leakage.
    # ---------------------------------------------------------
    device_customer_counts = {}
    ip_customer_counts = {}
    payment_customer_counts = {}

    device_customer_count = []
    ip_customer_count = []
    payment_customer_count = []

    for customer, device, ip, payment in zip(
        customer_key,
        device_key,
        ip_key,
        payment_key,
    ):
        device_customers = device_customer_counts.setdefault(
            device,
            set(),
        )

        ip_customers = ip_customer_counts.setdefault(
            ip,
            set(),
        )

        payment_customers = payment_customer_counts.setdefault(
            payment,
            set(),
        )

        device_customer_count.append(
            len(device_customers)
        )

        ip_customer_count.append(
            len(ip_customers)
        )

        payment_customer_count.append(
            len(payment_customers)
        )

        device_customers.add(customer)
        ip_customers.add(customer)
        payment_customers.add(customer)

    result["device_customer_count"] = device_customer_count
    result["ip_customer_count"] = ip_customer_count
    result["payment_customer_count"] = payment_customer_count

    # ---------------------------------------------------------
    # Time features
    # ---------------------------------------------------------
    seconds_in_day = 24 * 60 * 60

    result["transaction_day"] = (
        df["TransactionDT"] // seconds_in_day
    )

    result["transaction_hour"] = (
        (df["TransactionDT"] % seconds_in_day) // 3600
    )

    result["transaction_week"] = (
        df["TransactionDT"] // (7 * seconds_in_day)
    )

    # ---------------------------------------------------------
    # Missingness
    # ---------------------------------------------------------
    result["missing_count"] = df.isna().sum(axis=1)

    return result


def main() -> None:
    input_path = ARTIFACT_DIR / "train_combined.parquet"
    output_path = ARTIFACT_DIR / "train_online_features.parquet"

    print("Loading combined dataset...")

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

    print(
        f"Rows: {len(df):,}"
    )

    print("Building historical online features...")

    features = build_online_features(df)

    print(
        f"Output rows: {len(features):,}"
    )

    print(
        f"Output columns: {len(features.columns):,}"
    )

    print("\nFeature columns:")

    for column in features.columns:
        print(f"  {column}")

    print("\nFraud distribution:")

    print(
        features["isFraud"].value_counts()
    )

    print("\nSaving...")

    features.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()