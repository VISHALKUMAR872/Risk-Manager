from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CAL_PATH = ROOT / "artifacts" / "online_v5_priors_calibration.parquet"
TEST_PATH = ROOT / "artifacts" / "online_v5_priors_test.parquet"

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "fraud_online_v2_priors_lightgbm.txt"
)

CALIBRATOR_PATH = (
    ROOT / "artifacts" / "models"
    / "fraud_online_v2_priors_lightgbm_isotonic_calibrator.joblib"
)

REPORT_PATH = (
    ROOT / "artifacts" / "reports" / "v2"
    / "v2_lightgbm_policy_optimization.json"
)

FRONTIER_PATH = (
    ROOT / "artifacts" / "reports" / "v2"
    / "v2_lightgbm_policy_frontier.csv"
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


BEST_ITERATION = 122
LGF = 0.80

VERIFY_COST = 20.0
REVIEW_COST = 75.0
HOLD_COST = 150.0

FPR_BUDGETS = {
    "LOW_FRICTION": 0.0025,
    "BALANCED": 0.0100,
}


def evaluate_actions(
    y: np.ndarray,
    amount: np.ndarray,
    actions: np.ndarray,
    lgf: float = LGF,
) -> dict:

    fraud = y == 1
    legitimate = ~fraud
    intervention = actions > 0

    tp_mask = intervention & fraud
    fp_mask = intervention & legitimate

    tp = int(np.count_nonzero(tp_mask))
    fp = int(np.count_nonzero(fp_mask))
    tn = int(np.count_nonzero((~intervention) & legitimate))
    fn = int(np.count_nonzero((~intervention) & fraud))

    fraud_count = int(np.count_nonzero(fraud))
    legit_count = int(np.count_nonzero(legitimate))

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / fraud_count
        if fraud_count
        else 0.0
    )

    fpr = (
        fp / legit_count
        if legit_count
        else 0.0
    )

    verify = actions == 1
    review = actions == 2
    hold = actions == 3

    loss_avoided = float(
        amount[tp_mask].sum() * lgf
    )

    intervention_cost = float(
        np.count_nonzero(verify) * VERIFY_COST
        + np.count_nonzero(review) * REVIEW_COST
        + np.count_nonzero(hold) * HOLD_COST
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "intervention_rate": float(
            intervention.mean()
        ),
        "verify_count": int(
            np.count_nonzero(verify)
        ),
        "review_count": int(
            np.count_nonzero(review)
        ),
        "hold_count": int(
            np.count_nonzero(hold)
        ),
        "loss_avoided": loss_avoided,
        "intervention_cost": intervention_cost,
        "net_value": float(
            loss_avoided - intervention_cost
        ),
    }


def intervention_mask_from_rank(
    probability: np.ndarray,
    k: int,
) -> np.ndarray:

    """
    Select exactly the top-k transactions by calibrated
    fraud probability.

    This gives an exact intervention count and therefore
    an exact FPR budget control on the calibration partition.
    """

    n = len(probability)

    if k <= 0:
        return np.zeros(
            n,
            dtype=bool,
        )

    if k >= n:
        return np.ones(
            n,
            dtype=bool,
        )

    # argpartition is O(n), rather than repeatedly evaluating
    # threshold combinations.
    order = np.argpartition(
        probability,
        -k,
    )[-k:]

    mask = np.zeros(
        n,
        dtype=bool,
    )

    mask[order] = True

    return mask


def severity_allocation(
    intervention_mask: np.ndarray,
    expected_loss: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
) -> tuple[np.ndarray, dict]:

    """
    Within the already-selected intervention set:

        highest EL -> HOLD
        next EL    -> REVIEW
        remaining  -> VERIFY

    The number of REVIEW/HOLD transactions is optimized over a
    compact deterministic grid.

    This keeps intervention count fixed while optimizing the
    economics of customer friction.
    """

    indices = np.flatnonzero(
        intervention_mask
    )

    if len(indices) == 0:
        actions = np.zeros(
            len(expected_loss),
            dtype=np.int8,
        )

        return (
            actions,
            evaluate_actions(
                y,
                amount,
                actions,
            ),
        )

    order = indices[
        np.argsort(
            expected_loss[indices]
        )[::-1]
    ]

    n = len(order)

    # Candidate fractions for higher-friction actions.
    fractions = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.00,
                    0.10,
                    11,
                ),
                np.linspace(
                    0.125,
                    0.50,
                    16,
                ),
                np.linspace(
                    0.60,
                    1.00,
                    9,
                ),
            ]
        )
    )

    best_actions = None
    best_result = None

    for hold_fraction in fractions:

        hold_count = int(
            round(
                n * hold_fraction
            )
        )

        for review_fraction in fractions:

            review_count = int(
                round(
                    n * review_fraction
                )
            )

            if (
                hold_count
                + review_count
                > n
            ):
                continue

            actions = np.zeros(
                len(expected_loss),
                dtype=np.int8,
            )

            # Default intervention = VERIFY.
            actions[indices] = 1

            # Highest EL -> HOLD.
            if hold_count > 0:
                actions[
                    order[:hold_count]
                ] = 3

            # Next highest EL -> REVIEW.
            review_start = hold_count
            review_end = (
                hold_count
                + review_count
            )

            if review_end > review_start:
                actions[
                    order[
                        review_start:review_end
                    ]
                ] = 2

            result = evaluate_actions(
                y,
                amount,
                actions,
            )

            if (
                best_result is None
                or result["net_value"]
                > best_result["net_value"]
            ):
                best_actions = actions
                best_result = result

    if best_actions is None:
        raise RuntimeError(
            "Severity allocation produced no candidate."
        )

    return (
        best_actions,
        best_result,
    )


def optimize_fpr_frontier(
    y: np.ndarray,
    amount: np.ndarray,
    probability: np.ndarray,
    expected_loss: np.ndarray,
    fpr_budget: float,
) -> tuple[int, np.ndarray, dict, pd.DataFrame]:

    fraud = y == 1
    legitimate = ~fraud

    legit_count = int(
        np.count_nonzero(legitimate)
    )

    # Maximum number of legitimate transactions that may
    # be incorrectly intercepted.
    max_fp = int(
        np.floor(
            legit_count * fpr_budget
        )
    )

    print(
        f"  Legitimate transactions: "
        f"{legit_count:,}"
    )

    print(
        f"  Maximum allowed FP: "
        f"{max_fp:,}"
    )

    # ------------------------------------------------------------
    # Exact probability-ranked intervention frontier.
    #
    # Search around the allowed FP count. This avoids using an
    # arbitrary quantile grid.
    # ------------------------------------------------------------

    fraud_count = int(
        np.count_nonzero(fraud)
    )

    # Candidate intervention counts.
    #
    # We include:
    #   - the exact FPR boundary
    #   - slightly more conservative points
    #   - a broader frontier for economic comparison
    #
    # The final selected policy must still respect the budget.
    candidate_counts = set()

    for fp_target in range(
        max(
            0,
            max_fp - 40,
        ),
        max_fp + 1,
    ):

        # Number of interventions is not identical to FP count
        # because some selected transactions are fraud.
        #
        # Approximate the required intervention count using
        # prevalence, then inspect a local range.
        estimated_k = (
            fp_target
            + int(
                round(
                    fraud_count
                    * fpr_budget
                    / max(
                        fpr_budget,
                        1e-9,
                    )
                )
            )
        )

        for delta in range(
            -100,
            101,
            10,
        ):

            k = estimated_k + delta

            if 1 <= k <= len(y):
                candidate_counts.add(
                    k
                )

    # Also evaluate the exact maximum number of transactions
    # that can be selected while respecting the FP budget.
    #
    # We discover this directly from the sorted probability order.
    order = np.argsort(
        probability
    )[::-1]

    cumulative_fp = np.cumsum(
        legitimate[order]
    )

    valid_positions = np.flatnonzero(
        cumulative_fp <= max_fp
    )

    if len(valid_positions):
        exact_k = int(
            valid_positions[-1] + 1
        )

        for delta in range(
            -100,
            101,
        ):
            k = exact_k + delta

            if 1 <= k <= len(y):
                candidate_counts.add(
                    k
                )

    candidate_counts = sorted(
        candidate_counts
    )

    print(
        f"  Candidate intervention counts: "
        f"{len(candidate_counts)}"
    )

    best = None
    rows = []

    for k in candidate_counts:

        intervention = (
            intervention_mask_from_rank(
                probability,
                k,
            )
        )

        # Check the actual FPR before doing severity allocation.
        fp = int(
            np.count_nonzero(
                intervention & legitimate
            )
        )

        fpr = (
            fp / legit_count
        )

        if fpr > fpr_budget:
            continue

        actions, result = severity_allocation(
            intervention,
            expected_loss,
            y,
            amount,
        )

        row = {
            "intervention_count": k,
            "fpr_budget": fpr_budget,
            **result,
        }

        rows.append(
            row
        )

        if (
            best is None
            or result["net_value"]
            > best[2]["net_value"]
        ):
            best = (
                k,
                actions,
                result,
            )

    if best is None:
        raise RuntimeError(
            "No feasible probability-ranked "
            "policy found."
        )

    return (
        best[0],
        best[1],
        best[2],
        pd.DataFrame(rows),
    )


def derive_thresholds(
    probability: np.ndarray,
    expected_loss: np.ndarray,
    actions: np.ndarray,
) -> dict:

    thresholds = {}

    for action_number, name in [
        (1, "verify"),
        (2, "review"),
        (3, "hold"),
    ]:

        mask = actions >= action_number

        if not np.any(mask):
            thresholds[
                name
            ] = {
                "probability_min": None,
                "expected_loss_min": None,
            }
            continue

        thresholds[
            name
        ] = {
            "probability_min": float(
                probability[mask].min()
            ),
            "expected_loss_min": float(
                expected_loss[mask].min()
            ),
        }

    return thresholds


def print_policy(
    name: str,
    result: dict,
    thresholds: dict,
) -> None:

    print()
    print("-" * 72)
    print(name)
    print("-" * 72)

    print(
        "  Intervention threshold:"
    )

    print(
        f"    calibrated probability >= "
        f"{thresholds['intervention_probability']:.8f}"
    )

    print(
        "  Severity thresholds:"
    )

    for action in [
        "hold",
        "review",
        "verify",
    ]:

        info = thresholds[
            action
        ]

        value = info.get("expected_loss")

        if value is None:
            print(
                f"    {action.upper()}: "
                f"NONE — no transactions assigned"
            )
        else:
            print(
                f"    {action.upper()}: "
                f"EL >= "
                f"₹{value:.2f}"
            )

    print(
        f"  TP/FP/TN/FN: "
        f"{result['tp']}/"
        f"{result['fp']}/"
        f"{result['tn']}/"
        f"{result['fn']}"
    )

    print(
        f"  Precision: "
        f"{result['precision']:.4%}"
    )

    print(
        f"  Recall: "
        f"{result['recall']:.4%}"
    )

    print(
        f"  FPR: "
        f"{result['fpr']:.4%}"
    )

    print(
        f"  Intervention: "
        f"{result['intervention_rate']:.4%}"
    )

    print(
        f"  VERIFY/REVIEW/HOLD: "
        f"{result['verify_count']}/"
        f"{result['review_count']}/"
        f"{result['hold_count']}"
    )

    print(
        f"  Loss avoided: "
        f"₹{result['loss_avoided']:,.2f}"
    )

    print(
        f"  Intervention cost: "
        f"₹{result['intervention_cost']:,.2f}"
    )

    print(
        f"  Net value: "
        f"₹{result['net_value']:,.2f}"
    )


def main():

    print("=" * 72)
    print(
        "RISK SENTINEL V2 — EXACT FPR ECONOMIC FRONTIER"
    )
    print(
        "Calibration-only selection / locked future evaluation"
    )
    print("=" * 72)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration = pd.read_parquet(
        CAL_PATH
    )

    future_test = pd.read_parquet(
        TEST_PATH
    )

    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    if model.feature_name() != FEATURES:
        raise ValueError(
            "LightGBM feature order mismatch."
        )

    print()
    print(
        "GENERATING CALIBRATED PROBABILITIES"
    )

    p_cal = calibrator.predict(
        model.predict(
            calibration[FEATURES],
            num_iteration=BEST_ITERATION,
        )
    )

    p_test = calibrator.predict(
        model.predict(
            future_test[FEATURES],
            num_iteration=BEST_ITERATION,
        )
    )

    p_cal = np.asarray(
        p_cal,
        dtype=float,
    )

    p_test = np.asarray(
        p_test,
        dtype=float,
    )

    y_cal = calibration[
        "isFraud"
    ].to_numpy(dtype=int)

    y_test = future_test[
        "isFraud"
    ].to_numpy(dtype=int)

    amount_cal = calibration[
        "amount"
    ].to_numpy(dtype=float)

    amount_test = future_test[
        "amount"
    ].to_numpy(dtype=float)

    el_cal = (
        p_cal
        * amount_cal
        * LGF
    )

    el_test = (
        p_test
        * amount_test
        * LGF
    )

    print()
    print("DATA")
    print(
        f"  Calibration: "
        f"{len(y_cal):,}"
    )
    print(
        f"  Future test: "
        f"{len(y_test):,}"
    )
    print(
        f"  Calibration fraud: "
        f"{y_cal.sum():,}"
    )
    print(
        f"  Future fraud: "
        f"{y_test.sum():,}"
    )

    policies = {}

    for name, budget in [
        (
            "LOW_FRICTION",
            FPR_BUDGETS[
                "LOW_FRICTION"
            ],
        ),
        (
            "BALANCED",
            FPR_BUDGETS[
                "BALANCED"
            ],
        ),
    ]:

        print()
        print("=" * 72)
        print(
            f"CALIBRATION SEARCH — {name}"
        )
        print("=" * 72)

        (
            k,
            actions,
            result,
            frontier,
        ) = optimize_fpr_frontier(
            y_cal,
            amount_cal,
            p_cal,
            el_cal,
            budget,
        )

        intervention = actions > 0

        # Probability cutoff is the minimum probability
        # among selected transactions.
        intervention_probability = float(
            p_cal[intervention].min()
        )

        # Severity thresholds are based on EL among each
        # action class.
        hold_mask = actions == 3
        review_mask = actions == 2
        verify_mask = actions == 1

        severity_thresholds = {
            "verify": {
                "expected_loss": float(
                    el_cal[verify_mask].min()
                )
                if np.any(verify_mask)
                else None,
            },
            "review": {
                "expected_loss": float(
                    el_cal[review_mask].min()
                )
                if np.any(review_mask)
                else None,
            },
            "hold": {
                "expected_loss": float(
                    el_cal[hold_mask].min()
                )
                if np.any(hold_mask)
                else None,
            },
        }

        thresholds = {
            "intervention_probability":
                intervention_probability,
            **severity_thresholds,
        }

        policies[name] = {
            "intervention_count": k,
            "thresholds": thresholds,
            "calibration_result": result,
            "frontier": frontier,
        }

        print_policy(
            f"{name} — CALIBRATION",
            result,
            thresholds,
        )

    # ------------------------------------------------------------
    # Maximum recall under 1% FPR.
    # For this policy we directly maximize recall rather than
    # net value, while keeping severity costs visible.
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "CALIBRATION SEARCH — MAX RECALL UNDER 1% FPR"
    )
    print("=" * 72)

    fraud = y_cal == 1
    legitimate = ~fraud
    legit_count = int(
        legitimate.sum()
    )

    max_fp = int(
        np.floor(
            legit_count * 0.01
        )
    )

    order = np.argsort(
        p_cal
    )[::-1]

    cumulative_fp = np.cumsum(
        legitimate[order]
    )

    valid = np.flatnonzero(
        cumulative_fp <= max_fp
    )

    if len(valid) == 0:
        raise RuntimeError(
            "Unable to find 1% FPR operating point."
        )

    k = int(
        valid[-1] + 1
    )

    intervention = (
        intervention_mask_from_rank(
            p_cal,
            k,
        )
    )

    actions, result = severity_allocation(
        intervention,
        el_cal,
        y_cal,
        amount_cal,
    )

    intervention_probability = float(
        p_cal[intervention].min()
    )

    hold_mask = actions == 3
    review_mask = actions == 2
    verify_mask = actions == 1

    thresholds = {
        "intervention_probability":
            intervention_probability,
        "verify": {
            "expected_loss": float(
                el_cal[verify_mask].min()
            )
            if np.any(verify_mask)
            else None,
        },
        "review": {
            "expected_loss": float(
                el_cal[review_mask].min()
            )
            if np.any(review_mask)
            else None,
        },
        "hold": {
            "expected_loss": float(
                el_cal[hold_mask].min()
            )
            if np.any(hold_mask)
            else None,
        },
    }

    policies[
        "MAX_RECALL_UNDER_1PCT_FPR"
    ] = {
        "intervention_count": k,
        "thresholds": thresholds,
        "calibration_result": result,
        "frontier": None,
    }

    print_policy(
        "MAX_RECALL_UNDER_1PCT_FPR — CALIBRATION",
        result,
        thresholds,
    )

    # ------------------------------------------------------------
    # Locked future-test evaluation
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "LOCKED POLICIES — FUTURE TEST"
    )
    print("=" * 72)

    future_results = {}

    for name, policy in policies.items():

        threshold = policy[
            "thresholds"
        ]

        intervention = (
            p_test
            >= threshold[
                "intervention_probability"
            ]
        )

        actions = np.zeros(
            len(y_test),
            dtype=np.int8,
        )

        # All selected transactions start as VERIFY.
        actions[intervention] = 1

        # Apply severity thresholds.
        hold_el = threshold[
            "hold"
        ]["expected_loss"]

        review_el = threshold[
            "review"
        ]["expected_loss"]

        if hold_el is not None:
            actions[
                intervention
                & (el_test >= hold_el)
            ] = 3

        if review_el is not None:
            actions[
                intervention
                & (el_test >= review_el)
                & (actions != 3)
            ] = 2

        result = evaluate_actions(
            y_test,
            amount_test,
            actions,
        )

        future_results[
            name
        ] = {
            "thresholds": threshold,
            **result,
        }

        print()
        print("-" * 72)
        print(name)
        print("-" * 72)

        print(
            f"  Intervention probability >= "
            f"{threshold['intervention_probability']:.8f}"
        )

        print(
            f"  VERIFY/REVIEW/HOLD: "
            f"{result['verify_count']}/"
            f"{result['review_count']}/"
            f"{result['hold_count']}"
        )

        print(
            f"  TP/FP/TN/FN: "
            f"{result['tp']}/"
            f"{result['fp']}/"
            f"{result['tn']}/"
            f"{result['fn']}"
        )

        print(
            f"  Precision: "
            f"{result['precision']:.4%}"
        )

        print(
            f"  Recall: "
            f"{result['recall']:.4%}"
        )

        print(
            f"  FPR: "
            f"{result['fpr']:.4%}"
        )

        print(
            f"  Intervention: "
            f"{result['intervention_rate']:.4%}"
        )

        print(
            f"  Loss avoided: "
            f"₹{result['loss_avoided']:,.2f}"
        )

        print(
            f"  Intervention cost: "
            f"₹{result['intervention_cost']:,.2f}"
        )

        print(
            f"  Net value: "
            f"₹{result['net_value']:,.2f}"
        )

    # ------------------------------------------------------------
    # LGF sensitivity on locked BALANCED policy.
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "LGF SENSITIVITY — BALANCED"
    )
    print("=" * 72)

    # Evaluate the same locked BALANCED future-test actions
    # under different LGF assumptions. The actions remain fixed;
    # only loss avoided and net economic value change.

    balanced_threshold = policies[
        "BALANCED"
    ]["thresholds"]

    balanced_intervention = (
        p_test
        >= balanced_threshold[
            "intervention_probability"
        ]
    )

    balanced_actions = np.zeros(
        len(y_test),
        dtype=np.int8,
    )

    balanced_actions[
        balanced_intervention
    ] = 1

    hold_el = balanced_threshold[
        "hold"
    ]["expected_loss"]

    review_el = balanced_threshold[
        "review"
    ]["expected_loss"]

    if hold_el is not None:
        balanced_actions[
            balanced_intervention
            & (el_test >= hold_el)
        ] = 3

    if review_el is not None:
        balanced_actions[
            balanced_intervention
            & (el_test >= review_el)
            & (balanced_actions != 3)
        ] = 2

    lgf_rows = []

    for lgf in [
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]:

        result = evaluate_actions(
            y_test,
            amount_test,
            balanced_actions,
            lgf=lgf,
        )

        lgf_rows.append(
            {
                "lgf": lgf,
                **result,
            }
        )

        print(
            f"  LGF={lgf:.2f}: "
            f"precision={result['precision']:.4%}, "
            f"recall={result['recall']:.4%}, "
            f"FPR={result['fpr']:.4%}, "
            f"loss_avoided=₹{result['loss_avoided']:,.2f}, "
            f"net=₹{result['net_value']:,.2f}"
        )

    # Save frontier.
    # ------------------------------------------------------------

    frontier_frames = []

    for name, policy in policies.items():

        frontier = policy[
            "frontier"
        ]

        if frontier is None:
            continue

        frame = frontier.copy()

        frame.insert(
            0,
            "policy_family",
            name,
        )

        frontier_frames.append(
            frame
        )

    if frontier_frames:
        pd.concat(
            frontier_frames,
            ignore_index=True,
        ).to_csv(
            FRONTIER_PATH,
            index=False,
        )

    # ------------------------------------------------------------
    # Save report.
    # ------------------------------------------------------------

    report = {
        "experiment": (
            "V2 LightGBM exact FPR economic policy frontier"
        ),
        "method": (
            "probability-ranked exact-FPR intervention "
            "frontier with expected-loss severity allocation"
        ),
        "selection_partition": (
            "online_v5_priors_calibration.parquet"
        ),
        "evaluation_partition": (
            "online_v5_priors_test.parquet"
        ),
        "future_test_locked": True,
        "best_iteration": BEST_ITERATION,
        "lgf": LGF,
        "economic_assumptions": {
            "verify_cost": VERIFY_COST,
            "review_cost": REVIEW_COST,
            "hold_cost": HOLD_COST,
        },
        "fpr_budgets": FPR_BUDGETS,
        "calibration_rows": int(
            len(y_cal)
        ),
        "calibration_fraud": int(
            y_cal.sum()
        ),
        "future_test_rows": int(
            len(y_test)
        ),
        "future_test_fraud": int(
            y_test.sum()
        ),
        "selected_policies": {
            name: {
                "intervention_count": value[
                    "intervention_count"
                ],
                "thresholds": value[
                    "thresholds"
                ],
                "calibration_result": value[
                    "calibration_result"
                ],
            }
            for name, value in policies.items()
        },
        "future_test_results": future_results,
        "lgf_sensitivity": lgf_rows,
        "warning": (
            "LGF and intervention costs are illustrative "
            "assumptions. Future-test labels were not used "
            "for policy selection. The intervention frontier "
            "is probability-ranked; expected loss is used "
            "for severity allocation."
        ),
        "artifacts": {
            "report": str(
                REPORT_PATH
            ),
            "frontier": str(
                FRONTIER_PATH
            ),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "EXACT FPR POLICY OPTIMIZATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        f"Frontier: {FRONTIER_PATH}"
    )


if __name__ == "__main__":
    main()

