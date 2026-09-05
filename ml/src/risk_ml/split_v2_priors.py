from pathlib import Path
import pandas as pd

ARTIFACT_DIR = Path("artifacts")

SOURCE = ARTIFACT_DIR / "train_online_features_v5_priors.parquet"

PARTITIONS = {
    "train": ARTIFACT_DIR / "online_v5_priors_train.parquet",
    "calibration": ARTIFACT_DIR / "online_v5_priors_calibration.parquet",
    "test": ARTIFACT_DIR / "online_v5_priors_test.parquet",
}

print("=" * 80)
print("RISK SENTINEL V2 — SPLIT HISTORICAL-PRIOR ARTIFACT")
print("=" * 80)

df = pd.read_parquet(SOURCE)

print(f"Source rows: {len(df):,}")

train_ids = set(
    pd.read_parquet(
        ARTIFACT_DIR / "online_v4_train.parquet",
        columns=["TransactionID"],
    )["TransactionID"]
)

calibration_ids = set(
    pd.read_parquet(
        ARTIFACT_DIR / "online_v4_calibration.parquet",
        columns=["TransactionID"],
    )["TransactionID"]
)

test_ids = set(
    pd.read_parquet(
        ARTIFACT_DIR / "online_v4_test.parquet",
        columns=["TransactionID"],
    )["TransactionID"]
)

assert not train_ids & calibration_ids
assert not train_ids & test_ids
assert not calibration_ids & test_ids

partition_sets = {
    "train": train_ids,
    "calibration": calibration_ids,
    "test": test_ids,
}

for name, ids in partition_sets.items():
    part = df[df["TransactionID"].isin(ids)].copy()

    expected = len(ids)

    if len(part) != expected:
        raise RuntimeError(
            f"{name}: row mismatch {len(part)} != {expected}"
        )

    if part["TransactionID"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate TransactionID"
        )

    part.to_parquet(
        PARTITIONS[name],
        index=False,
    )

    print(
        f"{name.upper():12s}: "
        f"{len(part):,} rows, "
        f"{len(part.columns)} columns"
    )

print("\nChecking complete coverage...")

output_ids = set()

for path in PARTITIONS.values():
    output_ids.update(
        pd.read_parquet(
            path,
            columns=["TransactionID"],
        )["TransactionID"]
    )

source_ids = set(df["TransactionID"])

print("Source unique IDs:", len(source_ids))
print("Output unique IDs:", len(output_ids))
print("Missing IDs:", len(source_ids - output_ids))
print("Extra IDs:", len(output_ids - source_ids))

if source_ids != output_ids:
    raise RuntimeError("Partition coverage mismatch.")

print("\nV2 PARTITION BUILD: PASS")
