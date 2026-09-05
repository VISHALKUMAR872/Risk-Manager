from __future__ import annotations

"""
Risk Sentinel V6.1 — partition-integrity audit.

Purpose
-------
Verify that the calibration/selection partition used to choose V6.1 policy
thresholds is disjoint from the untouched future-test partition.

Primary key:
    TransactionID

Fallback:
    deterministic row fingerprint over the feature/target columns.

The audit also checks:
- row counts
- duplicate TransactionID values inside each partition
- exact TransactionID overlap
- fallback fingerprint overlap
- TransactionDT range overlap
- target counts
- amount totals
- a sample of overlapping IDs/fingerprints, if any

Exit code:
    0 = PASS, partitions are disjoint
    1 = FAIL, row-level overlap detected
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ARTIFACTS / "reports"

DEFAULT_SELECTION = ARTIFACTS / "online_v4_calibration.parquet"
DEFAULT_TEST = ARTIFACTS / "online_v4_test.parquet"

# Use the actual dataset identity plus model inputs/target for the fallback.
FINGERPRINT_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]


def stable_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def row_fingerprint(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in FINGERPRINT_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Cannot build fallback fingerprint; missing columns: {missing}"
        )

    canonical = df[FINGERPRINT_COLUMNS].map(stable_value).astype(str)
    joined = canonical.agg("\x1f".join, axis=1)
    return joined.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )


def summarize(name: str, df: pd.DataFrame) -> dict:
    result = {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    if "TransactionID" in df.columns:
        ids = df["TransactionID"]
        result["transaction_id_unique"] = int(ids.nunique(dropna=False))
        result["transaction_id_duplicates"] = int(ids.duplicated(keep=False).sum())
        result["transaction_id_nulls"] = int(ids.isna().sum())

    if "TransactionDT" in df.columns and len(df):
        result["transaction_dt_min"] = int(df["TransactionDT"].min())
        result["transaction_dt_max"] = int(df["TransactionDT"].max())

    if "isFraud" in df.columns:
        result["fraud_rows"] = int(df["isFraud"].sum())
        result["fraud_rate"] = float(df["isFraud"].mean())

    if "amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        result["amount_total"] = float(amount.fillna(0).sum())

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-path",
        type=str,
        default=str(DEFAULT_SELECTION),
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(DEFAULT_TEST),
    )
    args = parser.parse_args()

    selection_path = Path(args.selection_path)
    test_path = Path(args.test_path)

    if not selection_path.is_absolute():
        selection_path = ARTIFACTS / selection_path

    if not test_path.is_absolute():
        test_path = ARTIFACTS / test_path

    if not selection_path.exists():
        raise FileNotFoundError(selection_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    selection = pd.read_parquet(selection_path)
    test = pd.read_parquet(test_path)

    print("=" * 88)
    print("RISK SENTINEL — V6.1 PARTITION INTEGRITY AUDIT")
    print("=" * 88)
    print(f"Selection : {selection_path}")
    print(f"Future test: {test_path}")
    print()

    selection_summary = summarize("selection", selection)
    test_summary = summarize("future_test", test)

    print("PARTITION SUMMARY")
    print("-" * 88)
    print(json.dumps(
        {
            "selection": selection_summary,
            "future_test": test_summary,
        },
        indent=2,
    ))
    print()

    result = {
        "selection_partition": str(selection_path),
        "future_test_partition": str(test_path),
        "selection": selection_summary,
        "future_test": test_summary,
        "transaction_id_audit": {},
        "fingerprint_audit": {},
        "temporal_audit": {},
        "status": "FAIL",
    }

    # ------------------------------------------------------------------
    # Primary identity audit: TransactionID
    # ------------------------------------------------------------------
    if "TransactionID" not in selection.columns or "TransactionID" not in test.columns:
        raise RuntimeError(
            "TransactionID must exist in both partitions for the primary audit."
        )

    selection_ids = pd.Index(selection["TransactionID"].dropna().unique())
    test_ids = pd.Index(test["TransactionID"].dropna().unique())
    overlap_ids = selection_ids.intersection(test_ids)

    id_pass = len(overlap_ids) == 0

    result["transaction_id_audit"] = {
        "selection_unique_ids": int(len(selection_ids)),
        "future_test_unique_ids": int(len(test_ids)),
        "overlap_count": int(len(overlap_ids)),
        "overlap_sample": [int(x) for x in overlap_ids[:20]],
        "pass": id_pass,
    }

    print("TRANSACTIONID OVERLAP AUDIT")
    print("-" * 88)
    print(f"Selection unique IDs : {len(selection_ids):,}")
    print(f"Future-test unique IDs: {len(test_ids):,}")
    print(f"Overlap              : {len(overlap_ids):,}")
    if len(overlap_ids):
        print(f"Sample overlap IDs   : {[int(x) for x in overlap_ids[:20]]}")
    print(f"STATUS               : {'PASS' if id_pass else 'FAIL'}")
    print()

    # ------------------------------------------------------------------
    # Fallback exact-row fingerprint audit.
    # This is deliberately independent of TransactionID intersection.
    # ------------------------------------------------------------------
    selection_fp = row_fingerprint(selection)
    test_fp = row_fingerprint(test)

    selection_fps = pd.Index(selection_fp.unique())
    test_fps = pd.Index(test_fp.unique())
    overlap_fp = selection_fps.intersection(test_fps)

    fp_pass = len(overlap_fp) == 0

    result["fingerprint_audit"] = {
        "selection_unique_fingerprints": int(len(selection_fps)),
        "future_test_unique_fingerprints": int(len(test_fps)),
        "overlap_count": int(len(overlap_fp)),
        "overlap_sample": [str(x) for x in overlap_fp[:20]],
        "pass": fp_pass,
    }

    print("DETERMINISTIC FINGERPRINT OVERLAP AUDIT")
    print("-" * 88)
    print(f"Selection unique fingerprints : {len(selection_fps):,}")
    print(f"Future-test unique fingerprints: {len(test_fps):,}")
    print(f"Overlap                       : {len(overlap_fp):,}")
    if len(overlap_fp):
        print(f"Sample overlap fingerprints   : {[str(x) for x in overlap_fp[:5]]}")
    print(f"STATUS                        : {'PASS' if fp_pass else 'FAIL'}")
    print()

    # ------------------------------------------------------------------
    # Temporal audit.
    # This is diagnostic, not the partition-integrity criterion by itself.
    # Equal time ranges are not automatically leakage if IDs are disjoint.
    # ------------------------------------------------------------------
    if "TransactionDT" in selection.columns and "TransactionDT" in test.columns:
        sel_min = int(selection["TransactionDT"].min())
        sel_max = int(selection["TransactionDT"].max())
        test_min = int(test["TransactionDT"].min())
        test_max = int(test["TransactionDT"].max())

        temporal_overlap = not (
            sel_max < test_min or test_max < sel_min
        )

        result["temporal_audit"] = {
            "selection_min": sel_min,
            "selection_max": sel_max,
            "future_test_min": test_min,
            "future_test_max": test_max,
            "ranges_overlap": temporal_overlap,
            "note": (
                "Temporal range overlap is diagnostic only. "
                "The primary integrity criterion is row-level disjointness."
            ),
        }

        print("TEMPORAL RANGE AUDIT")
        print("-" * 88)
        print(f"Selection   : {sel_min:,} -> {sel_max:,}")
        print(f"Future test : {test_min:,} -> {test_max:,}")
        print(f"Ranges overlap: {temporal_overlap}")
        print()

    # ------------------------------------------------------------------
    # Final verdict.
    # Both identity checks must pass.
    # ------------------------------------------------------------------
    overall_pass = id_pass and fp_pass
    result["status"] = "PASS" if overall_pass else "FAIL"

    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / "v6_1_partition_integrity_audit.json"
    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    if overall_pass:
        print("V6.1 PARTITION INTEGRITY: PASS")
        print("No TransactionID or deterministic-row overlap was found.")
        print("The future-test partition remains row-level disjoint from policy selection.")
    else:
        print("V6.1 PARTITION INTEGRITY: FAIL")
        print("Do NOT promote V6.1 policy thresholds to production.")
        print("Rebuild the selection/test partitioning before re-running policy optimization.")
    print("=" * 88)
    print(f"Report: {report_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
