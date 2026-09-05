

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
FEATURES = ARTIFACTS / "train_online_features_v4.parquet"
TRAIN = ARTIFACTS / "online_v4_train.parquet"
CALIBRATION = ARTIFACTS / "online_v4_calibration.parquet"
TEST = ARTIFACTS / "online_v4_test.parquet"
REPORT_DIR = ARTIFACTS / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
FEATURE_COLUMNS = [
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


def safe_auc(y: pd.Series, x: pd.Series) -> float:
    if y.nunique() < 2 or x.nunique(dropna=False) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, x))
    except ValueError:
        return float("nan")


def safe_ap(y: pd.Series, x: pd.Series) -> float:
    if y.nunique() < 2 or x.nunique(dropna=False) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, x))
    except ValueError:
        return float("nan")


def quantile_bins(reference: pd.Series, candidate: pd.Series, bins: int = 10):
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    cand = pd.to_numeric(candidate, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if ref.empty or ref.nunique() < 2:
        return None, None

    edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None, None

    # Extend endpoints so values outside the reference range are retained.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
    cand_bins = pd.cut(cand, bins=edges, include_lowest=True)
    return ref_bins.value_counts(sort=False, normalize=True), cand_bins.value_counts(sort=False, normalize=True)


def psi(reference: pd.Series, candidate: pd.Series, bins: int = 10) -> float:
    ref_dist, cand_dist = quantile_bins(reference, candidate, bins)
    if ref_dist is None:
        return float("nan")

    cand_dist = cand_dist.reindex(ref_dist.index, fill_value=0.0)
    eps = 1e-6
    ref_p = np.clip(ref_dist.to_numpy(dtype=float), eps, None)
    cand_p = np.clip(cand_dist.to_numpy(dtype=float), eps, None)

    return float(np.sum((cand_p - ref_p) * np.log(cand_p / ref_p)))


def pct_change(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    if a == 0:
        return float("nan")
    return float((b - a) / abs(a))


def semantic_classification(feature: str, row: pd.Series) -> tuple[str, str]:
    """
    Classification is deliberately conservative.

    GREEN:
      The replay semantics are directly comparable to a production feature.

    YELLOW:
      The replay semantics are sound, but the IEEE-CIS identity/proxy is not
      equivalent to a production identifier.

    RED:
      The feature is structurally degenerate or its dataset proxy is known to
      encode the same entity used to define the relationship, making the
      intended graph signal invalid.
    """
    zero_rate = float(row["zero_rate"])
    unique = int(row["n_unique"])

    if feature == "payment_customer_count":
        return (
            "RED",
            "The IEEE-CIS payment proxy includes card1 (the customer proxy), so "
            "distinct-payment-to-customer graph cardinality collapses to ~1."
        )

    if feature == "merchant_transaction_count":
        return (
            "RED",
            "ProductCD is only a low-cardinality merchant proxy; it is not a "
            "real merchant identifier, so cumulative merchant counts are not "
            "production-semantic."
        )

    if feature in {"device_transactions_1h", "ip_transactions_1h",
                   "device_customer_count", "ip_customer_count"}:
        return (
            "YELLOW",
            "Temporal replay semantics are production-aligned, but IEEE-CIS "
            "DeviceInfo/IP proxies do not provide the same identity fidelity as "
            "real production device/IP identifiers."
        )

    if feature == "customer_degree":
        return (
            "YELLOW",
            "The cumulative graph-degree definition is production-aligned, but "
            "its quality depends on the dataset's proxy identities."
        )

    if feature in {"customer_transactions_1m", "customer_transactions_1h"}:
        return (
            "GREEN",
            "Event-time rolling customer velocity with strict prior-event "
            "semantics is directly comparable to the production Redis feature."
        )

    if feature == "amount":
        return (
            "GREEN",
            "Current-transaction amount is directly available and has no identity "
            "proxy dependency."
        )

    if zero_rate > 0.999 or unique <= 1:
        return (
            "RED",
            "Feature is effectively constant/degenerate in the replay dataset."
        )

    return (
        "YELLOW",
        "No direct semantic failure was detected, but the feature should be "
        "validated against production identity semantics before deployment."
    )


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return pd.read_parquet(path)


def main() -> None:
    print("=" * 88)
    print("RISK SENTINEL — V4 FEATURE DIAGNOSTICS")
    print("=" * 88)

    print("\nLoading temporal partitions...")
    train = load(TRAIN)
    calibration = load(CALIBRATION)
    test = load(TEST)

    print(f"Train:       {len(train):,}")
    print(f"Calibration: {len(calibration):,}")
    print(f"Future test: {len(test):,}")

    for name, df in [
        ("train", train),
        ("calibration", calibration),
        ("future_test", test),
    ]:
        missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    print("\nFraud rates:")
    for name, df in [
        ("train", train),
        ("calibration", calibration),
        ("future_test", test),
    ]:
        print(f"  {name:<12}: {df[TARGET].mean():.6%}")

    rows = []

    for feature in FEATURE_COLUMNS:
        x_train = pd.to_numeric(train[feature], errors="coerce")
        x_cal = pd.to_numeric(calibration[feature], errors="coerce")
        x_test = pd.to_numeric(test[feature], errors="coerce")

        y_train = train[TARGET].astype(int)
        y_cal = calibration[TARGET].astype(int)
        y_test = test[TARGET].astype(int)

        train_zero = float((x_train.fillna(0) == 0).mean())
        cal_zero = float((x_cal.fillna(0) == 0).mean())
        test_zero = float((x_test.fillna(0) == 0).mean())

        train_unique = int(x_train.nunique(dropna=True))
        cal_unique = int(x_cal.nunique(dropna=True))
        test_unique = int(x_test.nunique(dropna=True))

        train_auc = safe_auc(y_train, x_train.fillna(0))
        cal_auc = safe_auc(y_cal, x_cal.fillna(0))
        test_auc = safe_auc(y_test, x_test.fillna(0))

        train_ap = safe_ap(y_train, x_train.fillna(0))
        cal_ap = safe_ap(y_cal, x_cal.fillna(0))
        test_ap = safe_ap(y_test, x_test.fillna(0))

        row = {
            "feature": feature,
            "n_unique": train_unique,
            "train_zero_rate": train_zero,
            "cal_zero_rate": cal_zero,
            "test_zero_rate": test_zero,
            "train_mean": float(x_train.mean()),
            "cal_mean": float(x_cal.mean()),
            "test_mean": float(x_test.mean()),
            "train_std": float(x_train.std()),
            "cal_std": float(x_cal.std()),
            "test_std": float(x_test.std()),
            "train_auc": train_auc,
            "cal_auc": cal_auc,
            "test_auc": test_auc,
            "train_pr_auc": train_ap,
            "cal_pr_auc": cal_ap,
            "test_pr_auc": test_ap,
            "psi_train_to_cal": psi(x_train, x_cal),
            "psi_train_to_test": psi(x_train, x_test),
            "auc_test_minus_train": (
                test_auc - train_auc
                if np.isfinite(test_auc) and np.isfinite(train_auc)
                else float("nan")
            ),
            "pr_auc_test_minus_train": (
                test_ap - train_ap
                if np.isfinite(test_ap) and np.isfinite(train_ap)
                else float("nan")
            ),
        }

        # Fraud/non-fraud medians and means on the future period.
        fraud = x_test[y_test == 1]
        legit = x_test[y_test == 0]
        row["future_fraud_mean"] = float(fraud.mean())
        row["future_legit_mean"] = float(legit.mean())
        row["future_fraud_median"] = float(fraud.median())
        row["future_legit_median"] = float(legit.median())

        # A simple standardized separation score.
        pooled = math.sqrt(
            max(float(fraud.var()), 0.0) * max(len(fraud) - 1, 1)
            + max(float(legit.var()), 0.0) * max(len(legit) - 1, 1)
        )
        pooled_den = max(len(fraud) + len(legit) - 2, 1)
        pooled_sd = math.sqrt(max(pooled / pooled_den, 1e-12))
        row["future_mean_separation"] = float(
            abs(float(fraud.mean()) - float(legit.mean())) / pooled_sd
        )

        # Dataset/semantic assessment.
        semantic_row = pd.Series(
            {
                "zero_rate": test_zero,
                "n_unique": test_unique,
            }
        )
        level, reason = semantic_classification(feature, semantic_row)
        row["semantic_level"] = level
        row["semantic_reason"] = reason

        rows.append(row)

    report = pd.DataFrame(rows)

    # Correlation matrix.
    corr = train[FEATURE_COLUMNS].corr(method="spearman")
    corr.to_csv(REPORT_DIR / "v4_feature_spearman_correlation.csv")

    # Main report.
    report = report.sort_values(
        by=["semantic_level", "test_pr_auc"],
        ascending=[True, False],
        na_position="last",
    )
    report.to_csv(REPORT_DIR / "v4_feature_diagnostics.csv", index=False)

    # Drift summary.
    drift = report[
        [
            "feature",
            "psi_train_to_cal",
            "psi_train_to_test",
            "train_mean",
            "cal_mean",
            "test_mean",
            "train_zero_rate",
            "cal_zero_rate",
            "test_zero_rate",
        ]
    ].copy()
    drift["mean_change_train_to_test"] = [
        pct_change(a, b)
        for a, b in zip(drift["train_mean"], drift["test_mean"])
    ]
    drift.to_csv(REPORT_DIR / "v4_feature_drift.csv", index=False)

    # Compact JSON summary for automation/CI.
    summary = {
        "rows": {
            "train": len(train),
            "calibration": len(calibration),
            "future_test": len(test),
        },
        "fraud_rate": {
            "train": float(train[TARGET].mean()),
            "calibration": float(calibration[TARGET].mean()),
            "future_test": float(test[TARGET].mean()),
        },
        "features": report[
            [
                "feature",
                "semantic_level",
                "test_auc",
                "test_pr_auc",
                "psi_train_to_test",
            ]
        ].replace({np.nan: None}).to_dict(orient="records"),
        "semantic_counts": report["semantic_level"].value_counts().to_dict(),
    }
    (REPORT_DIR / "v4_feature_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("PER-FEATURE FUTURE-TEST DIAGNOSTICS")
    print("=" * 88)

    display_cols = [
        "feature",
        "semantic_level",
        "test_auc",
        "test_pr_auc",
        "psi_train_to_test",
        "test_zero_rate",
    ]
    print(
        report[display_cols].to_string(
            index=False,
            formatters={
                "test_auc": lambda x: f"{x:.4f}" if np.isfinite(x) else "NA",
                "test_pr_auc": lambda x: f"{x:.4f}" if np.isfinite(x) else "NA",
                "psi_train_to_test": lambda x: f"{x:.4f}" if np.isfinite(x) else "NA",
                "test_zero_rate": lambda x: f"{x:.2%}",
            },
        )
    )

    print("\n" + "=" * 88)
    print("DRIFT")
    print("=" * 88)

    drift_display = drift[
        ["feature", "psi_train_to_cal", "psi_train_to_test", "mean_change_train_to_test"]
    ]
    print(
        drift_display.to_string(
            index=False,
            formatters={
                "psi_train_to_cal": lambda x: f"{x:.4f}" if np.isfinite(x) else "NA",
                "psi_train_to_test": lambda x: f"{x:.4f}" if np.isfinite(x) else "NA",
                "mean_change_train_to_test": lambda x: (
                    f"{x:+.2%}" if np.isfinite(x) else "NA"
                ),
            },
        )
    )

    print("\n" + "=" * 88)
    print("SEMANTIC ASSESSMENT")
    print("=" * 88)

    for level in ["GREEN", "YELLOW", "RED"]:
        subset = report[report["semantic_level"] == level]
        print(f"\n{level}:")
        if subset.empty:
            print("  None")
        else:
            for _, r in subset.iterrows():
                print(f"  - {r['feature']}: {r['semantic_reason']}")

    print("\n" + "=" * 88)
    print("TOP FUTURE-TEST FEATURES BY PR-AUC")
    print("=" * 88)
    ranking = report.sort_values("test_pr_auc", ascending=False, na_position="last")
    for _, r in ranking.iterrows():
        print(
            f"  {r['feature']:<30} "
            f"PR-AUC={r['test_pr_auc']:.5f} "
            f"ROC-AUC={r['test_auc']:.5f}"
        )

    print("\nReports written to:")
    print(f"  {REPORT_DIR / 'v4_feature_diagnostics.csv'}")
    print(f"  {REPORT_DIR / 'v4_feature_drift.csv'}")
    print(f"  {REPORT_DIR / 'v4_feature_spearman_correlation.csv'}")
    print(f"  {REPORT_DIR / 'v4_feature_diagnostics_summary.json'}")

    print("\nV4 FEATURE DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()
