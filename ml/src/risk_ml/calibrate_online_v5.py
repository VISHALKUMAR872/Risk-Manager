from __future__ import annotations

from pathlib import Path

import joblib
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
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"

MODEL_PATH = MODEL_DIR / "fraud_online_v5_catboost.cbm"
CALIBRATION_PATH = ARTIFACTS / "online_v4_calibration.parquet"
TEST_PATH = ARTIFACTS / "online_v4_test.parquet"

CALIBRATOR_PATH = (
    MODEL_DIR / "fraud_online_v5_isotonic_calibrator.joblib"
)

TARGET = "isFraud"

V5_FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]


def metrics(
    y: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:

    return {
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "brier": float(
            brier_score_loss(y, probabilities)
        ),
    }


def main() -> None:

    print("=" * 88)
    print("RISK SENTINEL — V5 ISOTONIC CALIBRATION")
    print("=" * 88)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V5 model not found:\n{MODEL_PATH}"
        )

    print("\nLoading calibration and future-test datasets...")

    calibration = pd.read_parquet(
        CALIBRATION_PATH
    )

    future_test = pd.read_parquet(
        TEST_PATH
    )

    print(
        f"Calibration rows: {len(calibration):,}"
    )

    print(
        f"Future test rows: {len(future_test):,}"
    )

    # -------------------------------------------------------------------------
    # Validate features
    # -------------------------------------------------------------------------

    required = V5_FEATURES + [TARGET]

    for name, df in [
        ("calibration", calibration),
        ("future_test", future_test),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    print("\nLoading V5 CatBoost model...")

    model = CatBoostClassifier()

    model.load_model(
        MODEL_PATH
    )

    # -------------------------------------------------------------------------
    # Raw predictions
    # -------------------------------------------------------------------------

    X_cal = calibration[V5_FEATURES]
    y_cal = calibration[TARGET].astype(int)

    X_test = future_test[V5_FEATURES]
    y_test = future_test[TARGET].astype(int)

    raw_calibration = model.predict_proba(
        X_cal
    )[:, 1]

    raw_future = model.predict_proba(
        X_test
    )[:, 1]

    # -------------------------------------------------------------------------
    # Raw calibration-period metrics
    # -------------------------------------------------------------------------

    raw_cal_metrics = metrics(
        y_cal,
        raw_calibration,
    )

    print("\n" + "=" * 88)
    print("CALIBRATION PERIOD — RAW V5")
    print("=" * 88)

    print(
        f"ROC-AUC: {raw_cal_metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:  {raw_cal_metrics['pr_auc']:.6f}"
    )

    print(
        f"Brier:   {raw_cal_metrics['brier']:.6f}"
    )

    print("\nRaw probability range:")

    print(
        f"{raw_calibration.min():.6f}"
        f" -> "
        f"{raw_calibration.max():.6f}"
    )

    # -------------------------------------------------------------------------
    # Fit isotonic calibrator
    # -------------------------------------------------------------------------

    print("\nFitting isotonic calibrator...")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrated_calibration = calibrator.fit_transform(
        raw_calibration,
        y_cal,
    )

    calibrated_cal_metrics = metrics(
        y_cal,
        calibrated_calibration,
    )

    print(
        f"Calibrated calibration Brier: "
        f"{calibrated_cal_metrics['brier']:.6f}"
    )

    # -------------------------------------------------------------------------
    # Apply to untouched future test
    # -------------------------------------------------------------------------

    print(
        "\nEvaluating on untouched future test..."
    )

    calibrated_future = calibrator.predict(
        raw_future
    )

    raw_future_metrics = metrics(
        y_test,
        raw_future,
    )

    calibrated_future_metrics = metrics(
        y_test,
        calibrated_future,
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("FUTURE TEST — V5")
    print("=" * 88)

    print(
        f"Raw ROC-AUC:        "
        f"{raw_future_metrics['roc_auc']:.6f}"
    )

    print(
        f"Calibrated ROC-AUC: "
        f"{calibrated_future_metrics['roc_auc']:.6f}"
    )

    print(
        f"Raw PR-AUC:         "
        f"{raw_future_metrics['pr_auc']:.6f}"
    )

    print(
        f"Calibrated PR-AUC:  "
        f"{calibrated_future_metrics['pr_auc']:.6f}"
    )

    print(
        f"Raw Brier:          "
        f"{raw_future_metrics['brier']:.6f}"
    )

    print(
        f"Calibrated Brier:   "
        f"{calibrated_future_metrics['brier']:.6f}"
    )

    brier_improvement = (
        raw_future_metrics["brier"]
        - calibrated_future_metrics["brier"]
    )

    print(
        f"Brier improvement:  "
        f"{brier_improvement:.6f}"
    )

    # -------------------------------------------------------------------------
    # Probability distribution
    # -------------------------------------------------------------------------

    distribution = pd.DataFrame(
        {
            "raw": raw_future,
            "calibrated": calibrated_future,
        }
    )

    print("\nProbability distribution:")

    print(
        distribution.describe()
    )

    # -------------------------------------------------------------------------
    # Save calibrator
    # -------------------------------------------------------------------------

    joblib.dump(
        calibrator,
        CALIBRATOR_PATH,
    )

    print(
        f"\nCalibrator saved:\n"
        f"{CALIBRATOR_PATH}"
    )

    print("\n" + "=" * 88)
    print("V5 CALIBRATION COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()