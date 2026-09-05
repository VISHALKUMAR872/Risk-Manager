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


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"

MODEL_PATH = MODEL_DIR / "fraud_online_v4_catboost.cbm"
CALIBRATION_PATH = ARTIFACT_DIR / "online_v4_calibration.parquet"
TEST_PATH = ARTIFACT_DIR / "online_v4_test.parquet"

CALIBRATOR_PATH = (
    MODEL_DIR / "fraud_online_v4_isotonic_calibrator.joblib"
)

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

TARGET = "isFraud"


def main() -> None:
    print("=" * 80)
    print("RISK SENTINEL — V4 ISOTONIC CALIBRATION")
    print("=" * 80)

    calibration = pd.read_parquet(CALIBRATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    print(
        f"Calibration rows: {len(calibration):,}"
    )
    print(
        f"Future test rows: {len(test):,}"
    )

    print("\nLoading CatBoost model...")

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    # ------------------------------------------------------------------
    # Calibration period
    # ------------------------------------------------------------------

    calibration_raw = model.predict_proba(
        calibration[FEATURE_COLUMNS]
    )[:, 1]

    y_calibration = (
        calibration[TARGET]
        .to_numpy()
        .astype(np.int8)
    )

    print("\nCalibration raw score range:")
    print(
        f"{calibration_raw.min():.6f} -> "
        f"{calibration_raw.max():.6f}"
    )

    raw_calibration_brier = brier_score_loss(
        y_calibration,
        calibration_raw,
    )

    print(
        f"Raw calibration Brier: "
        f"{raw_calibration_brier:.6f}"
    )

    # ------------------------------------------------------------------
    # Fit isotonic calibrator ONLY on calibration period.
    # ------------------------------------------------------------------

    print("\nFitting isotonic calibrator...")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        calibration_raw,
        y_calibration,
    )

    calibrated_calibration = calibrator.predict(
        calibration_raw
    )

    calibrated_calibration_brier = brier_score_loss(
        y_calibration,
        calibrated_calibration,
    )

    print(
        f"Calibrated calibration Brier: "
        f"{calibrated_calibration_brier:.6f}"
    )

    # ------------------------------------------------------------------
    # FUTURE TEST — untouched until now.
    # ------------------------------------------------------------------

    print("\nEvaluating on untouched future test...")

    y_test = (
        test[TARGET]
        .to_numpy()
        .astype(np.int8)
    )

    raw_test = model.predict_proba(
        test[FEATURE_COLUMNS]
    )[:, 1]

    calibrated_test = calibrator.predict(
        raw_test
    )

    raw_auc = roc_auc_score(
        y_test,
        raw_test,
    )

    calibrated_auc = roc_auc_score(
        y_test,
        calibrated_test,
    )

    raw_pr = average_precision_score(
        y_test,
        raw_test,
    )

    calibrated_pr = average_precision_score(
        y_test,
        calibrated_test,
    )

    raw_brier = brier_score_loss(
        y_test,
        raw_test,
    )

    calibrated_brier = brier_score_loss(
        y_test,
        calibrated_test,
    )

    print("\n" + "=" * 80)
    print("FUTURE TEST — V4")
    print("=" * 80)

    print(
        f"Raw ROC-AUC:        {raw_auc:.6f}"
    )

    print(
        f"Calibrated ROC-AUC: {calibrated_auc:.6f}"
    )

    print(
        f"Raw PR-AUC:         {raw_pr:.6f}"
    )

    print(
        f"Calibrated PR-AUC:  {calibrated_pr:.6f}"
    )

    print(
        f"Raw Brier:          {raw_brier:.6f}"
    )

    print(
        f"Calibrated Brier:   {calibrated_brier:.6f}"
    )

    print(
        f"Brier improvement:  "
        f"{raw_brier - calibrated_brier:.6f}"
    )

    # ------------------------------------------------------------------
    # Probability distribution.
    # ------------------------------------------------------------------

    print("\nProbability distribution:")

    distribution = pd.DataFrame(
        {
            "raw": raw_test,
            "calibrated": calibrated_test,
        }
    ).describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )

    print(distribution.to_string())

    # ------------------------------------------------------------------
    # Save calibrator.
    # ------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        calibrator,
        CALIBRATOR_PATH,
    )

    print(
        f"\nCalibrator saved: "
        f"{CALIBRATOR_PATH}"
    )

    print("\n" + "=" * 80)
    print("V4 CALIBRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
