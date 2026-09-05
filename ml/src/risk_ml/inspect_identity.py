from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    path = DATA_DIR / "train_identity.csv"

    print("=" * 80)
    print("FILE: train_identity.csv")
    print(f"PATH: {path}")
    print(f"SIZE: {path.stat().st_size / (1024 ** 2):.2f} MB")

    df = pd.read_csv(path, nrows=5000)

    print(f"\nSample rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values - top 20:")
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(20)
        * 100
    )

    print(missing.to_string())

    print("\nTransactionID:")
    print(f"Unique: {df['TransactionID'].nunique():,}")
    print(f"Rows:   {len(df):,}")

    print("\nDuplicate TransactionID:")
    print(df["TransactionID"].duplicated().sum())


if __name__ == "__main__":
    main()
