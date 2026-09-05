from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "artifacts" / "models" / "fraud_online_v2_priors_catboost.cbm"

CALIBRATION_PATH = (
    ROOT / "artifacts" / "online_v5_priors_calibration.parquet"
)

TEST_PATH = (
    ROOT / "artifacts" / "online_v5_priors_test.parquet"
)

CALIBRATOR_PATH = (
    ROOT / "artifacts" / "models" / "fraud_online_v2_priors_isotonic_calibrator.joblib"
)

REPORT_PATH = (
    ROOT / "artifacts" / "reports" / "v2" / "v2_priors_calibration_evaluation.json"
)

CURVE_PATH = (
    ROOT / "artifacts" / "reports" / "v2" / "v2_priors_reliability.csv"
)

PLOT_PATH = (
    ROOT / "artifacts" / "reports" / "v2" / "v2_priors_reliability.png"
)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Calculate ECE using equal-width probability bins."""

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    rows: list[dict] = []
    total = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_probability": np.nan,
                    "observed_rate": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue

        mean_probability = float(probabilities[mask].mean())
        observed_rate = float(y_true[mask].mean())
        absolute_gap = abs(mean_probability - observed_rate)

        ece += (count / total) * absolute_gap

        rows.append(
            {
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": absolute_gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "mean_probability": float(probabilities.mean()),
        "std_probability": float(probabilities.std()),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
        "count_p_ge_0_90": int((probabilities >= 0.90).sum()),
        "count_p_ge_0_70": int((probabilities >= 0.70).sum()),
        "count_p_ge_0_40": int((probabilities >= 0.40).sum()),
        "count_p_eq_0": int((probabilities == 0.0).sum()),
        "count_p_eq_1": int((probabilities == 1.0).sum()),
    }


def main() -> None:
    print("=" * 72)
    print("RISK SENTINEL V2 PRIOR MODEL — ISOTONIC CALIBRATION")
    print("=" * 72)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    model_features = list(model.feature_names_)

    print(f"Model features: {len(model_features)}")
    print(f"Best iteration: {model.get_best_iteration()}")

    # ------------------------------------------------------------------
    # Load calibration and future-test partitions
    # ------------------------------------------------------------------

    calibration = pd.read_parquet(CALIBRATION_PATH)
    future_test = pd.read_parquet(TEST_PATH)

    missing_cal = [
        feature
        for feature in model_features
        if feature not in calibration.columns
    ]

    missing_test = [
        feature
        for feature in model_features
        if feature not in future_test.columns
    ]

    if missing_cal:
        raise ValueError(
            f"Calibration partition missing model features: {missing_cal}"
        )

    if missing_test:
        raise ValueError(
            f"Future-test partition missing model features: {missing_test}"
        )

    X_cal = calibration[model_features]
    y_cal = calibration["isFraud"].to_numpy(dtype=int)

    X_test = future_test[model_features]
    y_test = future_test["isFraud"].to_numpy(dtype=int)

    # ------------------------------------------------------------------
    # Raw probabilities
    # ------------------------------------------------------------------

    p_cal_raw = model.predict_proba(X_cal)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    print()
    print("Calibration partition:")
    print(f"  Rows:  {len(y_cal):,}")
    print(f"  Fraud: {int(y_cal.sum()):,}")
    print(f"  Rate:  {y_cal.mean():.6%}")

    print()
    print("Future-test partition:")
    print(f"  Rows:  {len(y_test):,}")
    print(f"  Fraud: {int(y_test.sum()):,}")
    print(f"  Rate:  {y_test.mean():.6%}")

    # ------------------------------------------------------------------
    # Fit isotonic ONLY on calibration partition
    # ------------------------------------------------------------------

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(p_cal_raw, y_cal)

    joblib.dump(calibrator, CALIBRATOR_PATH)

    # ------------------------------------------------------------------
    # Apply calibrator
    # ------------------------------------------------------------------

    p_calibrated = calibrator.predict(p_test_raw)

    # ------------------------------------------------------------------
    # Evaluate raw vs calibrated on untouched future test
    # ------------------------------------------------------------------

    raw_metrics = evaluate(y_test, p_test_raw)
    calibrated_metrics = evaluate(y_test, p_calibrated)

    raw_ece, raw_reliability = expected_calibration_error(
        y_test,
        p_test_raw,
    )

    calibrated_ece, calibrated_reliability = expected_calibration_error(
        y_test,
        p_calibrated,
    )

    raw_metrics["ece"] = raw_ece
    calibrated_metrics["ece"] = calibrated_ece

    reliability = calibrated_reliability.copy()
    reliability["raw_mean_probability"] = raw_reliability[
        "mean_probability"
    ]
    reliability["raw_observed_rate"] = raw_reliability[
        "observed_rate"
    ]

    reliability.to_csv(CURVE_PATH, index=False)

    # ------------------------------------------------------------------
    # Reliability plot
    # ------------------------------------------------------------------

    valid = reliability["count"] > 0

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        reliability.loc[valid, "mean_probability"],
        reliability.loc[valid, "observed_rate"],
        marker="o",
        label="V2 calibrated",
    )

    plt.plot(
        reliability.loc[valid, "raw_mean_probability"],
        reliability.loc[valid, "raw_observed_rate"],
        marker="o",
        label="V2 raw",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title("Risk Sentinel V2 — Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()

    # ------------------------------------------------------------------
    # Probability diagnostics
    # ------------------------------------------------------------------

    thresholds = {
        "p_ge_0_90": 0.90,
        "p_ge_0_70": 0.70,
        "p_ge_0_40": 0.40,
        "p_ge_0_25": 0.25,
        "p_ge_0_10": 0.10,
        "p_ge_0_05": 0.05,
    }

    threshold_counts = {
        key: int((p_calibrated >= threshold).sum())
        for key, threshold in thresholds.items()
    }

    # ------------------------------------------------------------------
    # Persist report
    # ------------------------------------------------------------------

    report = {
        "model": "fraud_online_v2_priors_catboost",
        "calibrator": "isotonic",
        "model_feature_count": len(model_features),
        "model_features": model_features,
        "best_iteration": int(model.get_best_iteration()),
        "calibration_rows": int(len(y_cal)),
        "calibration_fraud": int(y_cal.sum()),
        "calibration_fraud_rate": float(y_cal.mean()),
        "future_test_rows": int(len(y_test)),
        "future_test_fraud": int(y_test.sum()),
        "future_test_fraud_rate": float(y_test.mean()),
        "raw_future_test": raw_metrics,
        "calibrated_future_test": calibrated_metrics,
        "threshold_counts_calibrated": threshold_counts,
        "artifacts": {
            "calibrator": str(CALIBRATOR_PATH),
            "reliability_csv": str(CURVE_PATH),
            "reliability_plot": str(PLOT_PATH),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FUTURE-TEST RESULTS")
    print("=" * 72)

    print()
    print("RAW V2 MODEL")
    print(f"  ROC-AUC: {raw_metrics['roc_auc']:.6f}")
    print(f"  PR-AUC:  {raw_metrics['pr_auc']:.6f}")
    print(f"  Brier:   {raw_metrics['brier']:.6f}")
    print(f"  ECE:     {raw_metrics['ece']:.6f}")

    print()
    print("CALIBRATED V2 MODEL")
    print(f"  ROC-AUC: {calibrated_metrics['roc_auc']:.6f}")
    print(f"  PR-AUC:  {calibrated_metrics['pr_auc']:.6f}")
    print(f"  Brier:   {calibrated_metrics['brier']:.6f}")
    print(f"  ECE:     {calibrated_metrics['ece']:.6f}")

    print()
    print("CALIBRATED PROBABILITY DISTRIBUTION")
    print(
        f"  Mean: {calibrated_metrics['mean_probability']:.6f}"
    )
    print(
        f"  Std:  {calibrated_metrics['std_probability']:.6f}"
    )
    print(
        f"  Min:  {calibrated_metrics['min_probability']:.6f}"
    )
    print(
        f"  Max:  {calibrated_metrics['max_probability']:.6f}"
    )

    print()
    print("CALIBRATION ARTIFACT:")
    print(f"  {CALIBRATOR_PATH}")

    print()
    print("REPORT:")
    print(f"  {REPORT_PATH}")

    print()
    print("RELIABILITY DATA:")
    print(f"  {CURVE_PATH}")

    print()
    print("RELIABILITY PLOT:")
    print(f"  {PLOT_PATH}")

    print()
    print("V2 PRIOR MODEL CALIBRATION: COMPLETE")


if __name__ == "__main__":
    main()

