from pathlib import Path

import joblib
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"

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
    train_path = ARTIFACT_DIR / "online_v3_train.parquet"
    valid_path = ARTIFACT_DIR / "online_v3_valid.parquet"

    model_path = (
        MODEL_DIR
        / "fraud_online_v3_catboost.cbm"
    )

    calibrator_path = (
        MODEL_DIR
        / "fraud_online_v3_isotonic_calibrator.joblib"
    )

    print("Loading datasets...")

    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)

    # Use the last 25% of the training period for calibration.
    calibration_start = int(len(train) * 0.75)

    model_train = train.iloc[:calibration_start].copy()
    calibration = train.iloc[calibration_start:].copy()

    print("=" * 80)
    print("V3 CALIBRATION DATA")
    print("=" * 80)

    print(f"Model-training rows: {len(model_train):,}")
    print(f"Calibration rows:    {len(calibration):,}")
    print(f"Future validation:   {len(valid):,}")

    print("\nLoading CatBoost model...")

    model = CatBoostClassifier()
    model.load_model(model_path)

    print("\nGenerating calibration probabilities...")

    calibration_probability = model.predict_proba(
        calibration[FEATURE_COLUMNS]
    )[:, 1]

    calibration_target = (
        calibration[TARGET].to_numpy()
    )

    print(
        "Raw probability range: "
        f"{calibration_probability.min():.6f} -> "
        f"{calibration_probability.max():.6f}"
    )

    raw_brier = brier_score_loss(
        calibration_target,
        calibration_probability,
    )

    raw_auc = roc_auc_score(
        calibration_target,
        calibration_probability,
    )

    print(
        f"Raw Brier score: {raw_brier:.6f}"
    )

    print(
        f"Raw ROC-AUC: {raw_auc:.6f}"
    )

    print("\nFitting isotonic calibrator...")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        calibration_probability,
        calibration_target,
    )

    calibrated_probability = calibrator.predict(
        calibration_probability
    )

    calibrated_brier = brier_score_loss(
        calibration_target,
        calibrated_probability,
    )

    calibrated_auc = roc_auc_score(
        calibration_target,
        calibrated_probability,
    )

    print(
        f"Calibrated Brier score: "
        f"{calibrated_brier:.6f}"
    )

    print(
        f"Calibration improvement: "
        f"{raw_brier - calibrated_brier:.6f}"
    )

    print(
        f"Calibrated ROC-AUC: "
        f"{calibrated_auc:.6f}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        calibrator,
        calibrator_path,
    )

    print(
        f"\nCalibrator saved: "
        f"{calibrator_path}"
    )


if __name__ == "__main__":
    main()
