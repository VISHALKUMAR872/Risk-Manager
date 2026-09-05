from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def main() -> None:
    input_path = ARTIFACT_DIR / "train_features.parquet"

    train_path = ARTIFACT_DIR / "train_split.parquet"
    valid_path = ARTIFACT_DIR / "valid_split.parquet"

    print("Loading feature dataset...")

    df = pd.read_parquet(input_path)

    print(
        f"Rows: {len(df):,}"
    )

    print("Sorting by TransactionDT...")

    df = df.sort_values(
        "TransactionDT"
    ).reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    valid = df.iloc[split_index:].copy()

    print("=" * 80)
    print("TEMPORAL SPLIT")

    print(
        f"Train rows:      {len(train):,}"
    )

    print(
        f"Validation rows: {len(valid):,}"
    )

    print(
        f"Train time:      "
        f"{train['TransactionDT'].min():,} "
        f"-> "
        f"{train['TransactionDT'].max():,}"
    )

    print(
        f"Valid time:      "
        f"{valid['TransactionDT'].min():,} "
        f"-> "
        f"{valid['TransactionDT'].max():,}"
    )

    print("\nTrain fraud rate:")
    print(
        train["isFraud"]
        .mean()
        * 100
    )

    print("\nValidation fraud rate:")
    print(
        valid["isFraud"]
        .mean()
        * 100
    )

    print("\nSaving splits...")

    train.to_parquet(
        train_path,
        index=False,
    )

    valid.to_parquet(
        valid_path,
        index=False,
    )

    print(f"Saved: {train_path}")
    print(f"Saved: {valid_path}")


if __name__ == "__main__":
    main()
