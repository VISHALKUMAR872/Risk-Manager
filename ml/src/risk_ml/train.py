from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"


def main() -> None:
    train_path = ARTIFACT_DIR / "train_split.parquet"
    valid_path = ARTIFACT_DIR / "valid_split.parquet"

    print("Loading training data...")
    train = pd.read_parquet(train_path)

    print("Loading validation data...")
    valid = pd.read_parquet(valid_path)

    target = "isFraud"

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_valid = valid.drop(columns=[target])
    y_valid = valid[target]

    categorical_columns = X_train.select_dtypes(
        include=["str"]
    ).columns.tolist()

    print("=" * 80)
    print("CATBOOST TRAINING")
    print(f"Train rows: {len(X_train):,}")
    print(f"Valid rows: {len(X_valid):,}")
    print(f"Features: {len(X_train.columns):,}")
    print(f"Categorical features: {len(categorical_columns):,}")

    # TransactionDT is retained for temporal ordering but should
    # not be treated as a categorical variable.
    categorical_indices = [
        X_train.columns.get_loc(column)
        for column in categorical_columns
    ]

    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    print("\nTraining complete.")

    print(
        f"Best iteration: "
        f"{model.get_best_iteration()}"
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    print("\nVALIDATION METRICS")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC:  {pr_auc:.6f}")

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\nClassification report @ threshold 0.50:")
    print(
        classification_report(
            y_valid,
            predictions,
            digits=4,
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "fraud_catboost.cbm"

    model.save_model(model_path)

    print(f"\nModel saved: {model_path}")


if __name__ == "__main__":
    main()
