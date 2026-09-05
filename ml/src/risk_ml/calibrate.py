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


def main() -> None:
    train_path = ARTIFACT_DIR / "train_split.parquet"
    valid_path = ARTIFACT_DIR / "valid_split.parquet"
    model_path = MODEL_DIR / "fraud_catboost.cbm"

    print("Loading datasets...")

    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)

    target = "isFraud"

    # ---------------------------------------------------------
    # Split the original training period into:
    #
    # 75% model-training
    # 25% calibration
    # ---------------------------------------------------------
    calibration_index = int(len(train) * 0.75)

    model_train = train.iloc[:calibration_index].copy()
    calibration = train.iloc[calibration_index:].copy()

    X_calibration = calibration.drop(columns=[target])
    y_calibration = calibration[target]

    categorical_columns = X_calibration.select_dtypes(
        include=["str"]
    ).columns.tolist()

    categorical_indices = [
        X_calibration.columns.get_loc(column)
        for column in categorical_columns
    ]

    print("=" * 80)
    print("CALIBRATION DATA")
    print(f"Model-training rows: {len(model_train):,}")
    print(f"Calibration rows:    {len(calibration):,}")
    print(f"Validation rows:     {len(valid):,}")

    # ---------------------------------------------------------
    # Load the already-trained model.
    #
    # IMPORTANT:
    # This first calibration experiment uses the baseline model.
    # We are measuring how its probabilities behave.
    # ---------------------------------------------------------
    model = CatBoostClassifier()

    model.load_model(model_path)

    print("\nGenerating calibration probabilities...")

    raw_probabilities = model.predict_proba(
        X_calibration
    )[:, 1]

    print(
        f"Raw probability range: "
        f"{raw_probabilities.min():.6f} "
        f"-> "
        f"{raw_probabilities.max():.6f}"
    )

    raw_brier = brier_score_loss(
        y_calibration,
        raw_probabilities,
    )

    print(
        f"Raw Brier score: "
        f"{raw_brier:.6f}"
    )

    # ---------------------------------------------------------
    # Fit isotonic calibration.
    # ---------------------------------------------------------
    print("\nFitting isotonic calibrator...")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        raw_probabilities,
        y_calibration,
    )

    calibrated_probabilities = calibrator.predict(
        raw_probabilities
    )

    calibrated_brier = brier_score_loss(
        y_calibration,
        calibrated_probabilities,
    )

    print(
        f"Calibrated Brier score: "
        f"{calibrated_brier:.6f}"
    )

    print(
        f"Calibration improvement: "
        f"{raw_brier - calibrated_brier:.6f}"
    )

    # ROC-AUC should be essentially unchanged because
    # calibration should not fundamentally change ranking.
    raw_auc = roc_auc_score(
        y_calibration,
        raw_probabilities,
    )

    calibrated_auc = roc_auc_score(
        y_calibration,
        calibrated_probabilities,
    )

    print(
        f"\nRaw ROC-AUC: "
        f"{raw_auc:.6f}"
    )

    print(
        f"Calibrated ROC-AUC: "
        f"{calibrated_auc:.6f}"
    )

    # ---------------------------------------------------------
    # Save calibrator.
    # ---------------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibrator_path = (
        MODEL_DIR / "fraud_isotonic_calibrator.joblib"
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
