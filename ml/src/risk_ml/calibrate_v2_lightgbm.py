from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "artifacts" / "online_v5_priors_train.parquet"
CAL_PATH = ROOT / "artifacts" / "online_v5_priors_calibration.parquet"
TEST_PATH = ROOT / "artifacts" / "online_v5_priors_test.parquet"

MODEL_PATH = ROOT / "artifacts" / "models" / "fraud_online_v2_priors_lightgbm.txt"

CALIBRATOR_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "fraud_online_v2_priors_lightgbm_isotonic_calibrator.joblib"
)

REPORT_PATH = (
    ROOT
    / "artifacts"
    / "reports"
    / "v2"
    / "v2_lightgbm_calibration_evaluation.json"
)

RELIABILITY_PATH = (
    ROOT
    / "artifacts"
    / "reports"
    / "v2"
    / "v2_lightgbm_reliability.csv"
)

PLOT_PATH = (
    ROOT
    / "artifacts"
    / "reports"
    / "v2"
    / "v2_lightgbm_reliability.png"
)


FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
    "customer_historical_fraud_count",
    "customer_historical_transaction_count",
    "customer_historical_fraud_rate",
    "merchant_historical_fraud_count",
    "merchant_historical_transaction_count",
    "merchant_historical_fraud_rate",
    "device_historical_fraud_count",
    "device_historical_transaction_count",
    "device_historical_fraud_rate",
    "ip_historical_fraud_count",
    "ip_historical_transaction_count",
    "ip_historical_fraud_rate",
    "payment_historical_fraud_count",
    "payment_historical_transaction_count",
    "payment_historical_fraud_rate",
]


def ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    total = len(y_true)
    score = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

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
        gap = abs(mean_probability - observed_rate)

        score += (count / total) * gap

        rows.append(
            {
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": gap,
            }
        )

    return float(score), pd.DataFrame(rows)


def metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "mean_probability": float(probabilities.mean()),
        "std_probability": float(probabilities.std()),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
        "p_ge_0_90": int((probabilities >= 0.90).sum()),
        "p_ge_0_70": int((probabilities >= 0.70).sum()),
        "p_ge_0_40": int((probabilities >= 0.40).sum()),
        "p_ge_0_25": int((probabilities >= 0.25).sum()),
        "p_ge_0_10": int((probabilities >= 0.10).sum()),
        "p_ge_0_05": int((probabilities >= 0.05).sum()),
        "p_eq_0": int((probabilities == 0.0).sum()),
        "p_eq_1": int((probabilities == 1.0).sum()),
    }


def main() -> None:

    print("=" * 72)
    print("RISK SENTINEL V2 — LIGHTGBM ISOTONIC CALIBRATION")
    print("=" * 72)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load partitions
    # ---------------------------------------------------------------

    train = pd.read_parquet(TRAIN_PATH)
    calibration = pd.read_parquet(CAL_PATH)
    test = pd.read_parquet(TEST_PATH)

    for name, df in [
        ("train", train),
        ("calibration", calibration),
        ("future_test", test),
    ]:
        missing = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} missing features: {missing}"
            )

    X_train = train[FEATURES]
    y_train = train["isFraud"].to_numpy(dtype=int)

    X_cal = calibration[FEATURES]
    y_cal = calibration["isFraud"].to_numpy(dtype=int)

    X_test = test[FEATURES]
    y_test = test["isFraud"].to_numpy(dtype=int)

    print()
    print("DATA")
    print(f"  Train:       {len(train):,}")
    print(f"  Calibration: {len(calibration):,}")
    print(f"  Future test: {len(test):,}")
    print(f"  Features:    {len(FEATURES)}")
    print(f"  Test fraud:  {int(y_test.sum()):,}")
    print(f"  Test rate:   {y_test.mean():.6%}")

    # ---------------------------------------------------------------
    # Train LightGBM using calibration for early stopping
    # ---------------------------------------------------------------

    print()
    print("-" * 72)
    print("LIGHTGBM TRAINING")
    print("-" * 72)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        eval_X=X_cal,
        eval_y=y_cal,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=120,
                verbose=False,
            )
        ],
    )

    training_seconds = time.perf_counter() - start

    best_iteration = int(model.best_iteration_)

    # Save the exact LightGBM model used for calibration/evaluation.
    model.booster_.save_model(str(MODEL_PATH))

    print(f"  Best iteration: {best_iteration}")
    print(f"  Training time:  {training_seconds:.2f}s")

    # ---------------------------------------------------------------
    # Raw calibration probabilities
    # ---------------------------------------------------------------

    p_cal_raw = model.predict_proba(X_cal)[:, 1]

    # ---------------------------------------------------------------
    # Fit isotonic ONLY on calibration partition
    # ---------------------------------------------------------------

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        p_cal_raw,
        y_cal,
    )

    joblib.dump(
        calibrator,
        CALIBRATOR_PATH,
    )

    # ---------------------------------------------------------------
    # Future-test raw probabilities
    # ---------------------------------------------------------------

    start = time.perf_counter()

    p_test_raw = model.predict_proba(X_test)[:, 1]

    inference_seconds = time.perf_counter() - start

    # ---------------------------------------------------------------
    # Apply calibration to untouched future test
    # ---------------------------------------------------------------

    p_test_calibrated = calibrator.predict(
        p_test_raw
    )

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    raw = metrics(
        y_test,
        p_test_raw,
    )

    calibrated = metrics(
        y_test,
        p_test_calibrated,
    )

    raw_ece, raw_reliability = ece(
        y_test,
        p_test_raw,
    )

    calibrated_ece, calibrated_reliability = ece(
        y_test,
        p_test_calibrated,
    )

    raw["ece"] = raw_ece
    calibrated["ece"] = calibrated_ece

    # ---------------------------------------------------------------
    # Reliability data
    # ---------------------------------------------------------------

    reliability = calibrated_reliability.copy()

    reliability["raw_mean_probability"] = (
        raw_reliability["mean_probability"]
    )

    reliability["raw_observed_rate"] = (
        raw_reliability["observed_rate"]
    )

    reliability.to_csv(
        RELIABILITY_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Reliability diagram
    # ---------------------------------------------------------------

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
        label="LightGBM calibrated",
    )

    plt.plot(
        reliability.loc[valid, "raw_mean_probability"],
        reliability.loc[valid, "raw_observed_rate"],
        marker="o",
        label="LightGBM raw",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title("Risk Sentinel V2 — LightGBM Reliability")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        PLOT_PATH,
        dpi=160,
    )
    plt.close()

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {
        "experiment": "V2 LightGBM isotonic calibration",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "future_test_rows": int(len(test)),
        "future_test_fraud": int(y_test.sum()),
        "future_test_fraud_rate": float(y_test.mean()),
        "best_iteration": best_iteration,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "raw_future_test": raw,
        "calibrated_future_test": calibrated,
        "artifacts": {
            "model": str(MODEL_PATH),
            "calibrator": str(CALIBRATOR_PATH),
            "reliability_csv": str(RELIABILITY_PATH),
            "reliability_plot": str(PLOT_PATH),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("FUTURE-TEST RESULTS")
    print("=" * 72)

    print()
    print("RAW LIGHTGBM")
    print(f"  ROC-AUC: {raw['roc_auc']:.6f}")
    print(f"  PR-AUC:  {raw['pr_auc']:.6f}")
    print(f"  Brier:   {raw['brier']:.6f}")
    print(f"  ECE:     {raw['ece']:.6f}")

    print()
    print("CALIBRATED LIGHTGBM")
    print(f"  ROC-AUC: {calibrated['roc_auc']:.6f}")
    print(f"  PR-AUC:  {calibrated['pr_auc']:.6f}")
    print(f"  Brier:   {calibrated['brier']:.6f}")
    print(f"  ECE:     {calibrated['ece']:.6f}")

    print()
    print("CALIBRATED PROBABILITY DISTRIBUTION")
    print(f"  Mean: {calibrated['mean_probability']:.6f}")
    print(f"  Std:  {calibrated['std_probability']:.6f}")
    print(f"  Min:  {calibrated['min_probability']:.6f}")
    print(f"  Max:  {calibrated['max_probability']:.6f}")

    print()
    print("ARTIFACTS")
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Calibrator:  {CALIBRATOR_PATH}")
    print(f"  Report:      {REPORT_PATH}")
    print(f"  Reliability: {RELIABILITY_PATH}")
    print(f"  Plot:        {PLOT_PATH}")

    print()
    print("V2 LIGHTGBM CALIBRATION: COMPLETE")


if __name__ == "__main__":
    main()

