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


def main() -> None:
    model_path = MODEL_DIR / "fraud_catboost.cbm"
    calibrator_path = (
        MODEL_DIR / "fraud_isotonic_calibrator.joblib"
    )

    valid_path = ARTIFACT_DIR / "valid_split.parquet"

    print("Loading validation data...")
    valid = pd.read_parquet(valid_path)

    target = "isFraud"

    X_valid = valid.drop(columns=[target])
    y_valid = valid[target]

    categorical_columns = X_valid.select_dtypes(
        include=["str"]
    ).columns.tolist()

    categorical_indices = [
        X_valid.columns.get_loc(column)
        for column in categorical_columns
    ]

    print("Loading CatBoost model...")
    model = CatBoostClassifier()
    model.load_model(model_path)

    print("Loading calibrator...")
    calibrator = joblib.load(calibrator_path)

    print("Generating raw probabilities...")

    raw_probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    print("Generating calibrated probabilities...")

    calibrated_probabilities = calibrator.predict(
        raw_probabilities
    )

    raw_auc = roc_auc_score(
        y_valid,
        raw_probabilities,
    )

    calibrated_auc = roc_auc_score(
        y_valid,
        calibrated_probabilities,
    )

    raw_pr_auc = average_precision_score(
        y_valid,
        raw_probabilities,
    )

    calibrated_pr_auc = average_precision_score(
        y_valid,
        calibrated_probabilities,
    )

    raw_brier = brier_score_loss(
        y_valid,
        raw_probabilities,
    )

    calibrated_brier = brier_score_loss(
        y_valid,
        calibrated_probabilities,
    )

    print("=" * 80)
    print("FINAL FUTURE VALIDATION")
    print("=" * 80)

    print(f"Raw ROC-AUC:        {raw_auc:.6f}")
    print(f"Calibrated ROC-AUC: {calibrated_auc:.6f}")

    print(f"\nRaw PR-AUC:         {raw_pr_auc:.6f}")
    print(f"Calibrated PR-AUC:  {calibrated_pr_auc:.6f}")

    print(f"\nRaw Brier:          {raw_brier:.6f}")
    print(f"Calibrated Brier:   {calibrated_brier:.6f}")

    print(
        f"\nBrier improvement:  "
        f"{raw_brier - calibrated_brier:.6f}"
    )

    print("\nProbability statistics:")
    print(
        pd.Series(
            calibrated_probabilities
        ).describe().to_string()
    )

    # ---------------------------------------------------------
    # Evaluate Risk Sentinel policy thresholds.
    # ---------------------------------------------------------
    print("\nPOLICY THRESHOLD DISTRIBUTION")

    thresholds = [
        ("APPROVE", 0.00, 0.40),
        ("VERIFY", 0.40, 0.70),
        ("REVIEW", 0.70, 0.90),
        ("HOLD", 0.90, 1.01),
    ]

    total = len(calibrated_probabilities)

    for name, lower, upper in thresholds:
        mask = (
            (calibrated_probabilities >= lower)
            & (calibrated_probabilities < upper)
        )

        count = int(mask.sum())
        percentage = count / total * 100

        print(
            f"{name:8s} | "
            f"{count:7,d} | "
            f"{percentage:6.2f}%"
        )


if __name__ == "__main__":
    main()
