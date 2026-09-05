from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    transaction_path = DATA_DIR / "train_transaction.csv"
    identity_path = DATA_DIR / "train_identity.csv"

    print("Reading TransactionID columns...")

    transactions = pd.read_csv(
        transaction_path,
        usecols=["TransactionID"],
    )

    identity = pd.read_csv(
        identity_path,
        usecols=["TransactionID"],
    )

    transaction_ids = set(transactions["TransactionID"])
    identity_ids = set(identity["TransactionID"])

    matched = transaction_ids & identity_ids

    print("=" * 80)
    print("FULL DATASET RELATIONSHIP")
    print(f"Transaction rows: {len(transactions):,}")
    print(f"Identity rows:    {len(identity):,}")
    print(f"Matched IDs:      {len(matched):,}")

    transaction_match_rate = (
        len(matched) / len(transaction_ids) * 100
    )

    identity_match_rate = (
        len(matched) / len(identity_ids) * 100
    )

    print(
        f"Transaction rows with identity: "
        f"{transaction_match_rate:.2f}%"
    )

    print(
        f"Identity rows matching transactions: "
        f"{identity_match_rate:.2f}%"
    )

    print(
        f"Transaction-only rows: "
        f"{len(transaction_ids - identity_ids):,}"
    )

    print(
        f"Identity-only rows: "
        f"{len(identity_ids - transaction_ids):,}"
    )


if __name__ == "__main__":
    main()
