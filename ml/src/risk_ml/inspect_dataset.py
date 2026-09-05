from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def inspect_csv(filename: str, sample_rows: int = 5000) -> None:
    path = DATA_DIR / filename

    print("=" * 80)
    print(f"FILE: {filename}")
    print(f"PATH: {path}")
    print(f"SIZE: {path.stat().st_size / (1024 ** 2):.2f} MB")

    df = pd.read_csv(
        path,
        nrows=sample_rows,
    )

    print(f"\nSample rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nFirst 20 columns:")
    for column in df.columns[:20]:
        print(f"  {column}")

    print("\nData types:")
    print(df.dtypes.head(20))

    print("\nMissing values — top 20:")
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(20)
        * 100
    )

    print(missing.to_string())

    if "isFraud" in df.columns:
        print("\nFRAUD DISTRIBUTION:")
        print(df["isFraud"].value_counts(dropna=False))

        print("\nFRAUD PERCENTAGE:")
        print(
            df["isFraud"]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .to_string()
        )


def main() -> None:
    inspect_csv("train_transaction.csv")


if __name__ == "__main__":
    main()