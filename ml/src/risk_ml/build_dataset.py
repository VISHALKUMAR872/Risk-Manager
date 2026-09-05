from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    transaction_path = DATA_DIR / "train_transaction.csv"
    identity_path = DATA_DIR / "train_identity.csv"

    print("Loading transaction data...")

    transactions = pd.read_csv(transaction_path)

    print(
        f"Transactions loaded: "
        f"{len(transactions):,} rows, "
        f"{len(transactions.columns):,} columns"
    )

    print("Loading identity data...")

    identity = pd.read_csv(identity_path)

    print(
        f"Identity loaded: "
        f"{len(identity):,} rows, "
        f"{len(identity.columns):,} columns"
    )

    print("Merging on TransactionID...")

    identity_columns = [
        column
        for column in identity.columns
        if column != "TransactionID"
    ]

    combined = transactions.merge(
        identity[["TransactionID", *identity_columns]],
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Combined dataset: "
        f"{len(combined):,} rows, "
        f"{len(combined.columns):,} columns"
    )

    print("\nFraud distribution:")
    print(combined["isFraud"].value_counts())

    print("\nFraud percentage:")
    print(
        combined["isFraud"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nIdentity availability:")

    identity_available = combined["DeviceType"].notna()

    print(
        f"With identity: "
        f"{identity_available.sum():,}"
    )

    print(
        f"Without identity: "
        f"{(~identity_available).sum():,}"
    )

    print("\nSaving combined dataset...")

    output_path = OUTPUT_DIR / "train_combined.parquet"

    combined.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
