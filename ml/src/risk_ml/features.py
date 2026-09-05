from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


TARGET = "isFraud"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    # ---------------------------------------------------------
    # Core transaction features
    # ---------------------------------------------------------
    result["TransactionAmt"] = df["TransactionAmt"]
    result["TransactionDT"] = df["TransactionDT"]

    # Log amount reduces the influence of extreme transactions.
    result["TransactionAmt_log"] = (
        df["TransactionAmt"].clip(lower=0).add(1).map(
            lambda x: __import__("math").log(x)
        )
    )

    # ---------------------------------------------------------
    # Product / payment features
    # ---------------------------------------------------------
    categorical_columns = [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
    ]

    for column in categorical_columns:
        if column in df.columns:
            result[column] = df[column].fillna("__MISSING__").astype(str)

    # ---------------------------------------------------------
    # Card / address features
    # ---------------------------------------------------------
    numeric_columns = [
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
    ]

    for column in numeric_columns:
        if column in df.columns:
            result[column] = df[column]

    # ---------------------------------------------------------
    # Transaction count / amount features
    # ---------------------------------------------------------
    for column in [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
    ]:
        if column in df.columns:
            result[column] = df[column]

    # ---------------------------------------------------------
    # Time features
    # ---------------------------------------------------------
    # TransactionDT is elapsed time in seconds from an arbitrary
    # reference point in the IEEE-CIS dataset.
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
    # Identity features
    # ---------------------------------------------------------
    identity_numeric = [
        "id_01",
        "id_02",
        "id_03",
        "id_04",
        "id_05",
        "id_06",
        "id_07",
        "id_08",
        "id_09",
        "id_10",
        "id_11",
        "id_13",
        "id_14",
        "id_17",
        "id_18",
        "id_19",
        "id_20",
        "id_21",
        "id_22",
        "id_24",
        "id_25",
        "id_26",
        "id_32",
    ]

    for column in identity_numeric:
        if column in df.columns:
            result[column] = df[column]

    identity_categorical = [
        "id_12",
        "id_15",
        "id_16",
        "id_23",
        "id_27",
        "id_28",
        "id_29",
        "id_30",
        "id_31",
        "id_33",
        "id_34",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
    ]

    for column in identity_categorical:
        if column in df.columns:
            result[column] = (
                df[column]
                .fillna("__MISSING__")
                .astype(str)
            )

    for column in [
        "DeviceType",
        "DeviceInfo",
    ]:
        if column in df.columns:
            result[column] = (
                df[column]
                .fillna("__MISSING__")
                .astype(str)
            )

    # ---------------------------------------------------------
    # Missingness features
    # ---------------------------------------------------------
    result["missing_count"] = df.isna().sum(axis=1)

    result["identity_available"] = (
        df["DeviceType"].notna().astype(int)
    )

    return result


def main() -> None:
    input_path = ARTIFACT_DIR / "train_combined.parquet"
    output_path = ARTIFACT_DIR / "train_features.parquet"

    print("Loading combined dataset...")
    df = pd.read_parquet(input_path)

    print(
        f"Input: {len(df):,} rows, "
        f"{len(df.columns):,} columns"
    )

    print("Building features...")

    features = build_features(df)

    target = df[TARGET].astype("int8")

    features[TARGET] = target

    print(
        f"Output: {len(features):,} rows, "
        f"{len(features.columns):,} columns"
    )

    print("\nFeature dtypes:")
    print(features.dtypes.value_counts())

    print("\nFraud distribution:")
    print(features[TARGET].value_counts())

    print("\nSaving feature dataset...")

    features.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
