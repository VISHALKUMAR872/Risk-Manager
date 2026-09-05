from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def main() -> None:
    path = ARTIFACT_DIR / "train_combined.parquet"

    print("Loading combined dataset...")
    df = pd.read_parquet(path)

    print("=" * 80)
    print("COMBINED DATASET")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nTarget:")
    print(df["isFraud"].value_counts())
    print(
        df["isFraud"]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    print("\nData types:")
    print(df.dtypes.value_counts())

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = df.select_dtypes(
        exclude=["number"]
    ).columns

    print("\nNumeric columns:", len(numeric_columns))
    print("Categorical columns:", len(categorical_columns))

    print("\nTop 30 missing columns:")
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(30)
        .mul(100)
        .round(2)
    )

    print(missing.to_string())

    print("\nConstant columns:")
    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    print(f"Count: {len(constant_columns)}")

    for column in constant_columns:
        print(f"  {column}")

    print("\nTransaction time:")
    print(f"Min TransactionDT: {df['TransactionDT'].min():,}")
    print(f"Max TransactionDT: {df['TransactionDT'].max():,}")

    print("\nTransaction amount:")
    print(df["TransactionAmt"].describe().to_string())

    print("\nUnique values for important categorical columns:")

    for column in [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
    ]:
        if column in df.columns:
            print(
                f"\n{column}: "
                f"{df[column].nunique(dropna=True):,} unique"
            )
            print(
                df[column]
                .value_counts(dropna=False)
                .head(10)
                .to_string()
            )


if __name__ == "__main__":
    main()
