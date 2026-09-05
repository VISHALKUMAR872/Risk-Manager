from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from risk_engine.calibration import IsotonicCalibrator
from risk_engine.contracts import ExpectedLoss, FeatureVector, RiskScore
from risk_engine.expected_loss import ExpectedLossCalculator
from risk_engine.inference import CatBoostRiskModel, InferenceService
from risk_engine.policy import PolicyEngine


ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS = ROOT / "ml" / "artifacts"

MODEL_PATH = (
    ARTIFACTS
    / "models"
    / "fraud_online_v5_catboost.cbm"
)

CALIBRATOR_PATH = (
    ARTIFACTS
    / "models"
    / "fraud_online_v5_isotonic_calibrator.joblib"
)

FEATURE_ARTIFACT = (
    ARTIFACTS
    / "train_online_features_v4.parquet"
)

TEST_ARTIFACT = (
    ARTIFACTS
    / "online_v4_test.parquet"
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
]

TOLERANCE = 1e-10

TRANSACTION_DT_EPOCH = datetime(
    2017,
    1,
    1,
    tzinfo=timezone.utc,
)


def assert_close(
    name: str,
    expected: float,
    actual: float,
) -> None:
    if not np.isclose(
        expected,
        actual,
        rtol=1e-9,
        atol=TOLERANCE,
    ):
        raise AssertionError(
            f"{name} mismatch: "
            f"expected={expected:.15f}, "
            f"actual={actual:.15f}"
        )


def assert_equal(
    name: str,
    expected,
    actual,
) -> None:
    if expected != actual:
        raise AssertionError(
            f"{name} mismatch: "
            f"expected={expected!r}, "
            f"actual={actual!r}"
        )


def transaction_dt_to_datetime(
    transaction_dt: float,
) -> datetime:
    """
    Convert the offline relative TransactionDT value into a
    deterministic timezone-aware datetime.

    The absolute date is not a V5 model feature. This timestamp
    only satisfies the canonical FeatureVector contract.
    """

    return TRANSACTION_DT_EPOCH + timedelta(
        seconds=float(transaction_dt)
    )


def build_model_input(
    row: pd.Series,
) -> FeatureVector:
    """
    Construct the canonical FeatureVector consumed by the
    production inference path.
    """

    return FeatureVector(
        transaction_id=str(row["TransactionID"]),
        as_of_time=transaction_dt_to_datetime(
            row["TransactionDT"]
        ),
        amount=float(row["amount"]),
        currency="INR",
        customer_transactions_1m=int(
            row["customer_transactions_1m"]
        ),
        customer_transactions_1h=int(
            row["customer_transactions_1h"]
        ),
        device_transactions_1h=int(
            row["device_transactions_1h"]
        ),
        ip_transactions_1h=int(
            row["ip_transactions_1h"]
        ),
        customer_degree=int(
            row["customer_degree"]
        ),
        device_customer_count=int(
            row["device_customer_count"]
        ),
        ip_customer_count=int(
            row["ip_customer_count"]
        ),
        payment_customer_count=int(
            row.get(
                "payment_customer_count",
                0,
            )
        ),
        merchant_transaction_count=int(
            row.get(
                "merchant_transaction_count",
                0,
            )
        ),
    )


def build_risk_score(
    transaction_id: str,
    probability: float,
    model_version: str,
    calibration_version: str,
) -> RiskScore:
    return RiskScore(
        transaction_id=transaction_id,
        fraud_probability=probability,
        model_version=model_version,
        calibration_version=calibration_version,
    )


def main() -> None:
    print("=" * 88)
    print(
        "RISK SENTINEL — "
        "V5 MODEL / CALIBRATION / V6.1 POLICY PARITY GATE"
    )
    print("=" * 88)

    # ==================================================================
    # 1. Artifact checks
    # ==================================================================

    print("\nChecking artifacts...")

    for path in [
        MODEL_PATH,
        CALIBRATOR_PATH,
        FEATURE_ARTIFACT,
        TEST_ARTIFACT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

        print(
            f"  [PASS] {path.name}"
        )

    # ==================================================================
    # 2. Load production model + calibrator
    # ==================================================================

    print(
        "\nLoading production V5 model and calibrator..."
    )

    model = CatBoostRiskModel(
        MODEL_PATH,
    )

    calibrator = IsotonicCalibrator(
        CALIBRATOR_PATH,
    )

    inference = InferenceService(
        model=model,
        calibrator=calibrator,
    )

    print(
        "  [PASS] CatBoost model loaded"
    )

    print(
        "  [PASS] Isotonic calibrator loaded"
    )

    # ==================================================================
    # 3. Verify exact V5 model schema
    # ==================================================================

    expected_model_schema = FEATURE_COLUMNS

    actual_model_schema = model.feature_columns

    if actual_model_schema != expected_model_schema:
        raise AssertionError(
            "Production model feature schema mismatch:\n"
            f"expected={expected_model_schema}\n"
            f"actual={actual_model_schema}"
        )

    print(
        "  [PASS] Exact V5 feature schema verified"
    )

    print(
        f"       features = {actual_model_schema}"
    )

    # ==================================================================
    # 4. Load future-test data
    # ==================================================================

    print(
        "\nLoading future-test artifact..."
    )

    test = pd.read_parquet(
        TEST_ARTIFACT,
    )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in test.columns
    ]

    if missing_features:
        raise ValueError(
            "Future-test artifact missing features: "
            f"{missing_features}"
        )

    required_columns = [
        "TransactionID",
        "TransactionDT",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    if missing_required:
        raise ValueError(
            "Future-test artifact missing required columns: "
            f"{missing_required}"
        )

    print(
        f"Future-test rows: {len(test):,}"
    )

    # ==================================================================
    # 5. Deterministic checkpoints
    # ==================================================================

    checkpoints = []

    positions = np.linspace(
        0,
        len(test) - 1,
        20,
        dtype=int,
    )

    for position in positions:
        checkpoints.append(
            test.iloc[int(position)]
        )

    amount_indices = (
        test["amount"]
        .nlargest(10)
        .index
        .tolist()
    )

    for index in amount_indices:
        checkpoints.append(
            test.loc[index]
        )

    checkpoint_df = (
        pd.DataFrame(checkpoints)
        .drop_duplicates(
            subset=["TransactionID"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Deterministic checkpoints: "
        f"{len(checkpoint_df)}"
    )

    # ==================================================================
    # 6. Instantiate canonical production components
    # ==================================================================

    expected_loss_calculator = (
        ExpectedLossCalculator(
            loss_given_fraud=0.80,
        )
    )

    policy_engine = PolicyEngine()

    # ==================================================================
    # 7. V6.1 production policy contract
    # ==================================================================

    print(
        "\nChecking V6.1 production policy contract..."
    )

    assert_equal(
        "policy version",
        "policy-v6-balanced",
        policy_engine.VERSION,
    )

    config = policy_engine.config

    expected_thresholds = {
        "medium_probability": 0.275,
        "medium_expected_loss": 50.0,
        "high_probability": 0.560,
        "high_expected_loss": 400.0,
        "critical_probability": 0.640,
        "critical_expected_loss": 700.0,
    }

    for field, expected in expected_thresholds.items():
        actual = getattr(
            config,
            field,
        )

        assert_close(
            f"V6.1 {field}",
            expected,
            float(actual),
        )

    if not (
        config.medium_probability
        < config.high_probability
        < config.critical_probability
    ):
        raise AssertionError(
            "V6.1 probability thresholds "
            "are not strictly monotonic."
        )

    if not (
        config.medium_expected_loss
        < config.high_expected_loss
        < config.critical_expected_loss
    ):
        raise AssertionError(
            "V6.1 expected-loss thresholds "
            "are not strictly monotonic."
        )

    print(
        "  [PASS] policy version = policy-v6-balanced"
    )

    print(
        "  [PASS] VERIFY   : P>=0.275 OR EL>=50"
    )

    print(
        "  [PASS] REVIEW   : P>=0.560 OR EL>=400"
    )

    print(
        "  [PASS] HOLD     : P>=0.640 OR EL>=700"
    )

    print(
        "  [PASS] thresholds strictly monotonic"
    )

    # ==================================================================
    # 8. Model / calibration / expected-loss / policy parity
    # ==================================================================

    print(
        "\nRunning model/calibration/expected-loss/policy parity..."
    )

    verified = 0

    for _, row in checkpoint_df.iterrows():

        transaction_id = str(
            int(row["TransactionID"])
        )

        features = build_model_input(
            row
        )

        # --------------------------------------------------------------
        # Production inference path
        # --------------------------------------------------------------

        inference_result = inference.predict(
            features
        )

        if not hasattr(
            inference_result,
            "fraud_probability",
        ):
            raise TypeError(
                "InferenceService returned an unexpected "
                "RiskScore contract."
            )

        live_probability = float(
            inference_result.fraud_probability
        )

        # --------------------------------------------------------------
        # Direct production model inference
        # --------------------------------------------------------------

        raw_probability = float(
            model.predict_probability(
                features
            )
        )

        # --------------------------------------------------------------
        # Direct production calibration
        # --------------------------------------------------------------

        calibrated_probability = float(
            calibrator.calibrate(
                raw_probability
            )
        )

        # --------------------------------------------------------------
        # Compare production inference against direct
        # model + calibration calculation.
        # --------------------------------------------------------------

        assert_close(
            f"calibrated_probability "
            f"TransactionID={transaction_id}",
            calibrated_probability,
            live_probability,
        )

        assert_equal(
            f"model_version "
            f"TransactionID={transaction_id}",
            "fraud-online-v5",
            inference_result.model_version,
        )

        assert_equal(
            f"calibration_version "
            f"TransactionID={transaction_id}",
            calibrator.calibration_version,
            inference_result.calibration_version,
        )

        # --------------------------------------------------------------
        # Canonical RiskScore
        # --------------------------------------------------------------

        risk_score = build_risk_score(
            transaction_id=transaction_id,
            probability=calibrated_probability,
            model_version=inference_result.model_version,
            calibration_version=(
                inference_result.calibration_version
            ),
        )

        # --------------------------------------------------------------
        # Canonical expected-loss calculation
        # --------------------------------------------------------------

        amount = float(
            row["amount"]
        )

        currency = features.currency

        direct_expected_loss = (
            calibrated_probability
            * amount
            * 0.80
        )

        expected_loss_result = (
            expected_loss_calculator.calculate(
                risk_score=risk_score,
                exposure_amount=amount,
                currency=currency,
            )
        )

        calculated_expected_loss = float(
            expected_loss_result.expected_loss
        )

        assert_close(
            f"expected_loss "
            f"TransactionID={transaction_id}",
            direct_expected_loss,
            calculated_expected_loss,
        )

        assert_equal(
            f"expected_loss transaction_id "
            f"TransactionID={transaction_id}",
            transaction_id,
            expected_loss_result.transaction_id,
        )

        assert_equal(
            f"expected_loss currency "
            f"TransactionID={transaction_id}",
            currency,
            expected_loss_result.currency,
        )

        assert_equal(
            f"expected_loss version "
            f"TransactionID={transaction_id}",
            "expected-loss-v1",
            expected_loss_result.expected_loss_version,
        )

        # --------------------------------------------------------------
        # Canonical policy calculation
        # --------------------------------------------------------------

        direct_policy = policy_engine.decide(
            risk_score=risk_score,
            expected_loss=expected_loss_result,
        )

        # Re-run using an independently constructed equivalent
        # ExpectedLoss object to prove policy determinism.
        equivalent_expected_loss = ExpectedLoss(
            transaction_id=transaction_id,
            fraud_probability=calibrated_probability,
            exposure_amount=amount,
            loss_given_fraud=0.80,
            expected_loss=calculated_expected_loss,
            currency=currency,
            expected_loss_version="expected-loss-v1",
        )

        equivalent_policy = policy_engine.decide(
            risk_score=risk_score,
            expected_loss=equivalent_expected_loss,
        )

        assert_equal(
            f"policy decision "
            f"TransactionID={transaction_id}",
            direct_policy,
            equivalent_policy,
        )

        verified += 1

        print(
            f"[PASS] {verified:02d}/"
            f"{len(checkpoint_df):02d} "
            f"TransactionID={transaction_id} "
            f"raw={raw_probability:.6f} "
            f"calibrated={calibrated_probability:.6f} "
            f"loss={calculated_expected_loss:.2f} "
            f"decision={direct_policy.decision}"
        )

    # ==================================================================
    # 9. Policy threshold boundary tests
    # ==================================================================

    print(
        "\nTesting policy threshold boundaries..."
    )

    boundary_cases = [
        {
            "name": "very_low_risk",
            "probability": 0.001,
            "amount": 100.0,
        },
        {
            "name": "medium_probability",
            "probability": 0.40,
            "amount": 100.0,
        },
        {
            "name": "high_probability",
            "probability": 0.70,
            "amount": 100.0,
        },
        {
            "name": "critical_probability",
            "probability": 0.90,
            "amount": 100.0,
        },
        {
            "name": "high_expected_loss",
            "probability": 0.20,
            "amount": 5000.0,
        },
        {
            "name": "critical_expected_loss",
            "probability": 0.30,
            "amount": 5000.0,
        },
    ]

    for case in boundary_cases:

        probability = case["probability"]
        amount = case["amount"]

        risk_score = build_risk_score(
            transaction_id=(
                f"boundary-{case['name']}"
            ),
            probability=probability,
            model_version="fraud-online-v5",
            calibration_version=(
                calibrator.calibration_version
            ),
        )

        expected_loss = (
            expected_loss_calculator.calculate(
                risk_score=risk_score,
                exposure_amount=amount,
                currency="INR",
            )
        )

        first = policy_engine.decide(
            risk_score=risk_score,
            expected_loss=expected_loss,
        )

        second = policy_engine.decide(
            risk_score=risk_score,
            expected_loss=expected_loss,
        )

        assert_equal(
            case["name"],
            first,
            second,
        )

        print(
            f"  [PASS] {case['name']:<24} "
            f"P={probability:.3f} "
            f"loss={expected_loss.expected_loss:.2f} "
            f"decision={first.decision}"
        )

    # ==================================================================
    # 10. Explicit V6.1 probability bands
    # ==================================================================

    print(
        "\nTesting explicit V6.1 decision bands..."
    )

    explicit_boundary_cases = [
        {
            "name": "below_verify",
            "probability": 0.274999,
            "amount": 100.0,
            "expected_decision": "APPROVE",
        },
        {
            "name": "verify_probability",
            "probability": 0.275,
            "amount": 100.0,
            "expected_decision": "VERIFY",
        },
        {
            "name": "below_review",
            "probability": 0.559999,
            "amount": 100.0,
            "expected_decision": "VERIFY",
        },
        {
            "name": "review_probability",
            "probability": 0.560,
            "amount": 100.0,
            "expected_decision": "REVIEW",
        },
        {
            "name": "below_hold",
            "probability": 0.639999,
            "amount": 100.0,
            "expected_decision": "REVIEW",
        },
        {
            "name": "hold_probability",
            "probability": 0.640,
            "amount": 100.0,
            "expected_decision": "HOLD",
        },
    ]

    for case in explicit_boundary_cases:

        probability = case["probability"]
        amount = case["amount"]

        risk_score = build_risk_score(
            transaction_id=(
                f"probability-{case['name']}"
            ),
            probability=probability,
            model_version="fraud-online-v5",
            calibration_version=(
                calibrator.calibration_version
            ),
        )

        expected_loss = (
            expected_loss_calculator.calculate(
                risk_score=risk_score,
                exposure_amount=amount,
                currency="INR",
            )
        )

        result = policy_engine.decide(
            risk_score=risk_score,
            expected_loss=expected_loss,
        )

        actual_decision = result.decision

        if hasattr(
            actual_decision,
            "value",
        ):
            actual_decision = (
                actual_decision.value
            )

        actual_decision = str(
            actual_decision
        )

        expected_decision = (
            case["expected_decision"]
        )

        if actual_decision != expected_decision:
            raise AssertionError(
                f"{case['name']}: "
                f"expected decision={expected_decision}, "
                f"actual={actual_decision}, "
                f"P={probability}, "
                f"expected_loss="
                f"{expected_loss.expected_loss}"
            )

        print(
            f"  [PASS] {case['name']:<24} "
            f"P={probability:.6f} "
            f"loss={expected_loss.expected_loss:.2f} "
            f"decision={actual_decision}"
        )

    # ==================================================================
    # 11. Explicit expected-loss bands
    # ==================================================================

    print(
        "\nTesting explicit expected-loss boundaries..."
    )

    expected_loss_boundary_cases = [
        {
            "name": "below_verify_loss",
            "probability": 0.10,
            "amount": 624.99,
            "expected_decision": "APPROVE",
        },
        {
            "name": "verify_loss_boundary",
            "probability": 0.10,
            "amount": 625.00,
            "expected_decision": "VERIFY",
        },
        {
            "name": "below_review_loss",
            "probability": 0.10,
            "amount": 4999.99,
            "expected_decision": "VERIFY",
        },
        {
            "name": "review_loss_boundary",
            "probability": 0.10,
            "amount": 5000.00,
            "expected_decision": "REVIEW",
        },
        {
            "name": "below_hold_loss",
            "probability": 0.20,
            "amount": 4374.99,
            "expected_decision": "REVIEW",
        },
        {
            "name": "hold_loss_boundary",
            "probability": 0.20,
            "amount": 4375.00,
            "expected_decision": "HOLD",
        },
    ]

    for case in expected_loss_boundary_cases:

        probability = case["probability"]
        amount = case["amount"]

        risk_score = build_risk_score(
            transaction_id=(
                f"loss-{case['name']}"
            ),
            probability=probability,
            model_version="fraud-online-v5",
            calibration_version=(
                calibrator.calibration_version
            ),
        )

        expected_loss = (
            expected_loss_calculator.calculate(
                risk_score=risk_score,
                exposure_amount=amount,
                currency="INR",
            )
        )

        result = policy_engine.decide(
            risk_score=risk_score,
            expected_loss=expected_loss,
        )

        actual_decision = result.decision

        if hasattr(
            actual_decision,
            "value",
        ):
            actual_decision = (
                actual_decision.value
            )

        actual_decision = str(
            actual_decision
        )

        expected_decision = (
            case["expected_decision"]
        )

        if actual_decision != expected_decision:
            raise AssertionError(
                f"{case['name']}: "
                f"expected decision={expected_decision}, "
                f"actual={actual_decision}, "
                f"P={probability}, "
                f"expected_loss="
                f"{expected_loss.expected_loss}"
            )

        print(
            f"  [PASS] {case['name']:<24} "
            f"P={probability:.6f} "
            f"loss={expected_loss.expected_loss:.2f} "
            f"decision={actual_decision}"
        )

    # ==================================================================
    # 12. Inference determinism
    # ==================================================================

    print(
        "\nTesting inference determinism..."
    )

    if checkpoint_df.empty:
        raise AssertionError(
            "No checkpoints available."
        )

    deterministic_row = (
        checkpoint_df.iloc[0]
    )

    features = build_model_input(
        deterministic_row
    )

    outputs = []

    for _ in range(5):

        result = inference.predict(
            features
        )

        outputs.append(
            float(
                result.fraud_probability
            )
        )

    if len(set(outputs)) != 1:
        raise AssertionError(
            "Inference is not deterministic: "
            f"{outputs}"
        )

    print(
        f"  [PASS] repeated probability = "
        f"{outputs[0]:.12f}"
    )

    # ==================================================================
    # Final result
    # ==================================================================

    print(
        "\n" + "=" * 88
    )

    print(
        "V5 MODEL / CALIBRATION / "
        "V6.1 POLICY PARITY: PASS"
    )

    print(
        "=" * 88
    )

    print(
        f"Model/calibration checkpoints: "
        f"{verified}"
    )

    print(
        "Expected-loss formula: "
        "P(fraud) × amount × 0.80"
    )

    print(
        "Production policy: "
        "policy-v6-balanced"
    )

    print(
        "Policy engine: deterministic"
    )


if __name__ == "__main__":
    main()