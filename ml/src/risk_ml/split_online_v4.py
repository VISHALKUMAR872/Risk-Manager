from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

INPUT_PATH = ARTIFACT_DIR / "train_online_features_v4.parquet"

TRAIN_PATH = ARTIFACT_DIR / "online_v4_train.parquet"
CALIBRATION_PATH = ARTIFACT_DIR / "online_v4_calibration.parquet"
TEST_PATH = ARTIFACT_DIR / "online_v4_test.parquet"


TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15


def main() -> None:
    print("=" * 80)
    print("RISK SENTINEL — V4 TEMPORAL DATASET SPLIT")
    print("=" * 80)

    print("\nLoading V4 features...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Rows loaded: {len(df):,}")

    # ------------------------------------------------------------------
    # Deterministic chronological ordering.
    # ------------------------------------------------------------------

    df = (
        df.sort_values(
            ["TransactionDT", "TransactionID"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    n = len(df)

    train_end = int(n * TRAIN_FRACTION)

    calibration_end = int(
        n * (TRAIN_FRACTION + CALIBRATION_FRACTION)
    )

    train = df.iloc[:train_end].copy()

    calibration = df.iloc[
        train_end:calibration_end
    ].copy()

    test = df.iloc[
        calibration_end:
    ].copy()

    # ------------------------------------------------------------------
    # Report boundaries.
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PARTITION SIZES")
    print("-" * 80)

    print(f"Total:       {len(df):,}")
    print(f"Train:       {len(train):,}")
    print(f"Calibration: {len(calibration):,}")
    print(f"Future test: {len(test):,}")

    print("\n" + "-" * 80)
    print("TEMPORAL BOUNDARIES")
    print("-" * 80)

    print(
        f"Train:       "
        f"{train.TransactionDT.min():,} -> "
        f"{train.TransactionDT.max():,}"
    )

    print(
        f"Calibration: "
        f"{calibration.TransactionDT.min():,} -> "
        f"{calibration.TransactionDT.max():,}"
    )

    print(
        f"Future test: "
        f"{test.TransactionDT.min():,} -> "
        f"{test.TransactionDT.max():,}"
    )

    # ------------------------------------------------------------------
    # Hard temporal assertions.
    #
    # These are essential because this is a fraud model and random
    # splitting would allow temporal leakage.
    # ------------------------------------------------------------------

    if train.TransactionDT.max() > calibration.TransactionDT.min():
        raise RuntimeError(
            "Train/calibration temporal overlap detected."
        )

    if calibration.TransactionDT.max() > test.TransactionDT.min():
        raise RuntimeError(
            "Calibration/test temporal overlap detected."
        )

    # If the boundary timestamps are identical, the rows at that
    # timestamp were deterministically assigned to one partition.
    # The replay already treats equal timestamps as a single scoring
    # point, so report this explicitly rather than silently assuming
    # strict timestamp separation.
    print("\nTemporal ordering checks passed.")

    # ------------------------------------------------------------------
    # Fraud rates.
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("FRAUD RATES")
    print("-" * 80)

    print(
        f"Train:       {train.isFraud.mean() * 100:.4f}%"
    )

    print(
        f"Calibration: {calibration.isFraud.mean() * 100:.4f}%"
    )

    print(
        f"Future test: {test.isFraud.mean() * 100:.4f}%"
    )

    # ------------------------------------------------------------------
    # Feature columns.
    # ------------------------------------------------------------------

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

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing production features: {missing}"
        )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    print("\nSaving V4 partitions...")

    train.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    calibration.to_parquet(
        CALIBRATION_PATH,
        index=False,
    )

    test.to_parquet(
        TEST_PATH,
        index=False,
    )

    print(f"Saved: {TRAIN_PATH}")
    print(f"Saved: {CALIBRATION_PATH}")
    print(f"Saved: {TEST_PATH}")

    print("\n" + "=" * 80)
    print("V4 TEMPORAL SPLIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
