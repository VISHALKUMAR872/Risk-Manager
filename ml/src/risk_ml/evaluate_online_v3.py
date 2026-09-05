from pathlib import Path

import joblib
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
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
    valid_path = ARTIFACT_DIR / "online_v3_valid.parquet"

    model_path = (
        MODEL_DIR / "fraud_online_v3_catboost.cbm"
    )

    calibrator_path = (
        MODEL_DIR
        / "fraud_online_v3_isotonic_calibrator.joblib"
    )

    print("Loading future validation data...")
    valid = pd.read_parquet(valid_path)

    print("Loading CatBoost model...")
    model = CatBoostClassifier()
    model.load_model(model_path)

    print("Loading isotonic calibrator...")
    calibrator = joblib.load(calibrator_path)

    print("Generating raw probabilities...")

    raw_probability = model.predict_proba(
        valid[FEATURE_COLUMNS]
    )[:, 1]

    print("Generating calibrated probabilities...")

    calibrated_probability = calibrator.predict(
        raw_probability
    )

    y = valid[TARGET].to_numpy()

    raw_auc = roc_auc_score(y, raw_probability)
    calibrated_auc = roc_auc_score(y, calibrated_probability)

    raw_pr = average_precision_score(y, raw_probability)
    calibrated_pr = average_precision_score(
        y,
        calibrated_probability,
    )

    raw_brier = brier_score_loss(y, raw_probability)
    calibrated_brier = brier_score_loss(
        y,
        calibrated_probability,
    )

    print("=" * 80)
    print("V3 FINAL FUTURE VALIDATION")
    print("=" * 80)

    print(f"Rows: {len(valid):,}")

    print()
    print(f"Raw ROC-AUC:        {raw_auc:.6f}")
    print(f"Calibrated ROC-AUC: {calibrated_auc:.6f}")

    print()
    print(f"Raw PR-AUC:         {raw_pr:.6f}")
    print(f"Calibrated PR-AUC:  {calibrated_pr:.6f}")

    print()
    print(f"Raw Brier:          {raw_brier:.6f}")
    print(f"Calibrated Brier:   {calibrated_brier:.6f}")

    print()
    print(
        f"Brier improvement: "
        f"{raw_brier - calibrated_brier:.6f}"
    )

    print()
    print("Probability statistics:")

    print(
        pd.Series(calibrated_probability)
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()
