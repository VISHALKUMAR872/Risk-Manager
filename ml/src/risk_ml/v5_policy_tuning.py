from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"

CALIBRATION_PATH = ARTIFACTS / "online_v4_calibration.parquet"
MODEL_PATH = MODELS / "fraud_online_v5_catboost.cbm"
CALIBRATOR_PATH = MODELS / "fraud_online_v5_isotonic_calibrator.joblib"

LOCKED_POLICY_PATH = REPORTS / "v5_locked_policies.json"
TUNING_REPORT_PATH = REPORTS / "v5_policy_tuning.csv"
TUNING_SUMMARY_PATH = REPORTS / "v5_policy_tuning_summary.json"

LGF = 0.80

FEATURES = [
    "amount",
    "customer_transactions_1m",
    "customer_transactions_1h",
    "device_transactions_1h",
    "ip_transactions_1h",
    "customer_degree",
    "device_customer_count",
    "ip_customer_count",
]


# These are evaluation constraints for selecting operating points.
# They are NOT derived from the future test set.
MAX_LEGITIMATE_CHALLENGE = {
    "conservative": 0.0010,  # 0.10%
    "balanced": 0.0025,      # 0.25%
    "sensitive": 0.0050,     # 0.50%
}


VERIFY_PROBABILITIES = np.round(
    np.arange(0.10, 0.501, 0.05), 2
)

REVIEW_PROBABILITIES = np.round(
    np.arange(0.30, 0.801, 0.05), 2
)

HOLD_PROBABILITIES = np.round(
    np.arange(0.60, 0.951, 0.05), 2
)

VERIFY_EXPECTED_LOSSES = [
    50.0,
    75.0,
    100.0,
    125.0,
    150.0,
]

REVIEW_EXPECTED_LOSSES = [
    300.0,
    400.0,
    500.0,
    600.0,
    750.0,
]

HOLD_EXPECTED_LOSSES = [
    700.0,
    800.0,
    1000.0,
    1200.0,
    1500.0,
]


def apply_policy(
    probability: np.ndarray,
    expected_loss: np.ndarray,
    policy: dict,
) -> np.ndarray:

    decision = np.full(
        len(probability),
        "APPROVE",
        dtype=object,
    )

    verify = (
        (probability >= policy["verify_probability"])
        | (expected_loss >= policy["verify_expected_loss"])
    )

    review = (
        (probability >= policy["review_probability"])
        | (expected_loss >= policy["review_expected_loss"])
    )

    hold = (
        (probability >= policy["hold_probability"])
        | (expected_loss >= policy["hold_expected_loss"])
    )

    decision[verify] = "VERIFY"
    decision[review] = "REVIEW"
    decision[hold] = "HOLD"

    return decision


def evaluate_policy(
    decision: np.ndarray,
    probability: np.ndarray,
    amount: np.ndarray,
    actual: np.ndarray,
) -> dict:

    fraud_total = int(actual.sum())
    legitimate_total = len(actual) - fraud_total

    challenged = decision != "APPROVE"
    approved = ~challenged

    fraud_challenged = int(actual[challenged].sum())

    legitimate_challenged = int(
        (1 - actual[challenged]).sum()
    )

    baseline_loss = float(
        np.sum(actual * amount * LGF)
    )

    residual_loss = float(
        np.sum(actual[approved] * amount[approved] * LGF)
    )

    loss_avoided = baseline_loss - residual_loss

    return {
        "transactions": len(actual),
        "fraud_transactions": fraud_total,
        "legitimate_transactions": legitimate_total,

        "challenged_count": int(challenged.sum()),
        "challenge_rate": float(challenged.mean()),

        "fraud_challenged": fraud_challenged,
        "fraud_capture_rate": (
            fraud_challenged / fraud_total
            if fraud_total
            else 0.0
        ),

        "legitimate_challenged": legitimate_challenged,
        "legitimate_challenge_rate": (
            legitimate_challenged / legitimate_total
            if legitimate_total
            else 0.0
        ),

        "challenge_precision": (
            fraud_challenged / int(challenged.sum())
            if challenged.any()
            else 0.0
        ),

        "baseline_actual_loss": baseline_loss,
        "residual_actual_loss": residual_loss,
        "actual_loss_avoided": loss_avoided,
        "loss_avoidance_rate": (
            loss_avoided / baseline_loss
            if baseline_loss
            else 0.0
        ),

        "mean_probability": float(probability.mean()),
    }


def main() -> None:

    print("=" * 88)
    print("RISK SENTINEL — V5 CALIBRATION-ONLY POLICY TUNING")
    print("=" * 88)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading calibration partition...")

    df = pd.read_parquet(
        CALIBRATION_PATH
    )

    print(
        f"Calibration rows: {len(df):,}"
    )

    required = FEATURES + ["isFraud", "amount"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    print("\nLoading V5 model...")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    print("\nLoading V5 isotonic calibrator...")

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    X = df[FEATURES]

    y = (
        df["isFraud"]
        .astype(int)
        .to_numpy()
    )

    amount = (
        df["amount"]
        .astype(float)
        .to_numpy()
    )

    raw_probability = model.predict_proba(
        X
    )[:, 1]

    probability = np.asarray(
        calibrator.predict(raw_probability),
        dtype=float,
    )

    expected_loss = (
        probability
        * amount
        * LGF
    )

    print(
        f"\nCalibration fraud rate: "
        f"{y.mean():.4%}"
    )

    print(
        "\nSearching candidate policies..."
    )

    candidates = []

    for verify_p in VERIFY_PROBABILITIES:
        for review_p in REVIEW_PROBABILITIES:
            if review_p <= verify_p:
                continue

            for hold_p in HOLD_PROBABILITIES:
                if hold_p <= review_p:
                    continue

                for verify_loss in VERIFY_EXPECTED_LOSSES:
                    for review_loss in REVIEW_EXPECTED_LOSSES:

                        if review_loss <= verify_loss:
                            continue

                        for hold_loss in HOLD_EXPECTED_LOSSES:

                            if hold_loss <= review_loss:
                                continue

                            policy = {
                                "verify_probability": float(
                                    verify_p
                                ),
                                "review_probability": float(
                                    review_p
                                ),
                                "hold_probability": float(
                                    hold_p
                                ),
                                "verify_expected_loss": float(
                                    verify_loss
                                ),
                                "review_expected_loss": float(
                                    review_loss
                                ),
                                "hold_expected_loss": float(
                                    hold_loss
                                ),
                            }

                            decision = apply_policy(
                                probability,
                                expected_loss,
                                policy,
                            )

                            metrics = evaluate_policy(
                                decision,
                                probability,
                                amount,
                                y,
                            )

                            candidates.append(
                                {
                                    **policy,
                                    **metrics,
                                }
                            )

    candidates_df = pd.DataFrame(
        candidates
    )

    print(
        f"Candidate policies evaluated: "
        f"{len(candidates_df):,}"
    )

    locked = {}

    for operating_point, limit in (
        MAX_LEGITIMATE_CHALLENGE.items()
    ):

        eligible = candidates_df[
            candidates_df[
                "legitimate_challenge_rate"
            ] <= limit
        ].copy()

        if eligible.empty:
            raise RuntimeError(
                f"No eligible {operating_point} "
                f"policy under legitimate challenge "
                f"limit {limit:.4%}."
            )

        # Primary objective:
        # maximize actual loss avoided.
        #
        # Secondary:
        # maximize fraud capture.
        #
        # Tertiary:
        # maximize challenge precision.
        #
        # Final tie-breaker:
        # minimize challenge rate.

        eligible = eligible.sort_values(
            [
                "actual_loss_avoided",
                "fraud_capture_rate",
                "challenge_precision",
                "challenge_rate",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )

        best = eligible.iloc[0]

        locked[operating_point] = {
            "verify_probability": float(
                best["verify_probability"]
            ),
            "review_probability": float(
                best["review_probability"]
            ),
            "hold_probability": float(
                best["hold_probability"]
            ),
            "verify_expected_loss": float(
                best["verify_expected_loss"]
            ),
            "review_expected_loss": float(
                best["review_expected_loss"]
            ),
            "hold_expected_loss": float(
                best["hold_expected_loss"]
            ),
            "max_legitimate_challenge_rate": float(
                limit
            ),

            # Calibration-period diagnostics.
            "calibration_loss_avoided": float(
                best["actual_loss_avoided"]
            ),
            "calibration_loss_avoidance_rate": float(
                best["loss_avoidance_rate"]
            ),
            "calibration_fraud_capture_rate": float(
                best["fraud_capture_rate"]
            ),
            "calibration_challenge_precision": float(
                best["challenge_precision"]
            ),
            "calibration_legitimate_challenge_rate": float(
                best["legitimate_challenge_rate"]
            ),
        }

    # Save the complete candidate search for auditability.
    candidates_df.to_csv(
        TUNING_REPORT_PATH,
        index=False,
    )

    summary = {
        "dataset": "online_v4_calibration.parquet",
        "dataset_role": "CALIBRATION_ONLY",
        "future_test_used_for_selection": False,
        "loss_given_fraud": LGF,
        "constraints": {
            key: value
            for key, value in (
                MAX_LEGITIMATE_CHALLENGE.items()
            )
        },
        "locked_policies": locked,
    }

    TUNING_SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOCKED_POLICY_PATH.write_text(
        json.dumps(
            {
                "model": "fraud-online-v5",
                "calibrator": "isotonic-online-v5",
                "loss_given_fraud": LGF,
                "selection_dataset": (
                    "online_v4_calibration.parquet"
                ),
                "future_test_used_for_selection": False,
                "policies": locked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("LOCKED POLICIES")
    print("=" * 88)

    for name, policy in locked.items():

        print(f"\n{name.upper()}")

        print(
            f"  VERIFY: "
            f"P>={policy['verify_probability']:.2f} "
            f"OR EL>={policy['verify_expected_loss']:.0f}"
        )

        print(
            f"  REVIEW: "
            f"P>={policy['review_probability']:.2f} "
            f"OR EL>={policy['review_expected_loss']:.0f}"
        )

        print(
            f"  HOLD:   "
            f"P>={policy['hold_probability']:.2f} "
            f"OR EL>={policy['hold_expected_loss']:.0f}"
        )

        print(
            f"  Calibration fraud capture: "
            f"{policy['calibration_fraud_capture_rate']:.2%}"
        )

        print(
            f"  Calibration legit challenge: "
            f"{policy['calibration_legitimate_challenge_rate']:.2%}"
        )

        print(
            f"  Calibration loss avoided: "
            f"{policy['calibration_loss_avoided']:,.2f}"
        )

    print("\nArtifacts written:")

    print(
        f"  {TUNING_REPORT_PATH}"
    )

    print(
        f"  {TUNING_SUMMARY_PATH}"
    )

    print(
        f"  {LOCKED_POLICY_PATH}"
    )

    print("\nV5 CALIBRATION-ONLY POLICY TUNING COMPLETE")


if __name__ == "__main__":
    main()
