# Risk Sentinel

> **Real-time fraud loss prevention for merchants — turning transaction risk into economically-aware decisions.**

Risk Sentinel is a real-time fraud-risk platform combining **calibrated machine learning, transaction velocity, entity relationships, expected loss, and deterministic policy evaluation** to decide how a transaction should be handled.

Instead of treating fraud detection as a binary `fraud / not-fraud` classification problem, Risk Sentinel asks:

> **How risky is this transaction, how much could it cost, and what action is economically justified?**

**Hackathon Track:** AI Risk Manager / Fraud-Driven Transaction Loss Prevention

---

## Table of Contents

- [Executive Snapshot](#executive-snapshot)
- [1. The Problem](#1-the-problem)
- [2. Our Solution](#2-our-solution)
- [3. Why Risk Sentinel Is Different](#3-why-risk-sentinel-is-different)
- [4. System Architecture](#4-system-architecture)
- [5. End-to-End Data Flow](#5-end-to-end-data-flow)
- [6. AI / ML Architecture](#6-ai--ml-architecture)
- [7. Leakage-Safe Historical Features](#7-leakage-safe-historical-features)
- [8. V2 Feature Set](#8-v2-feature-set)
- [9. Model Benchmark](#9-model-benchmark)
- [10. Model Calibration](#10-model-calibration)
- [11. Expected Loss Decisioning](#11-expected-loss-decisioning)
- [12. Policy Optimization](#12-policy-optimization)
- [13. Locked Future Evaluation](#13-locked-future-evaluation)
- [14. Economic Evaluation](#14-economic-evaluation)
- [15. LGF Sensitivity](#15-lgf-sensitivity)
- [16. Streaming Benchmark](#16-streaming-benchmark)
- [17. Reliability](#17-reliability)
- [18. Explainability](#18-explainability)
- [19. API](#19-api)
- [20. Repository Structure](#20-repository-structure)
- [21. Local Setup](#21-local-setup)
- [22. Model Artifacts](#22-model-artifacts)
- [23. Implemented vs Planned](#23-implemented-vs-planned)
- [24. Limitations](#24-limitations)
- [25. Security & Production Hardening](#25-security--production-hardening)
- [26. Scalability](#26-scalability)
- [27. Observability](#27-observability)
- [28. Hackathon Track Alignment](#28-hackathon-track-alignment)
- [29. Why This Matters](#29-why-this-matters)

---

## Executive Snapshot

| Metric | Result |
|---|---:|
| Production V2 Model | **LightGBM** |
| V2 Model Features | **23** |
| Future ROC-AUC | **0.840223** |
| Future PR-AUC | **0.283188** |
| Calibrated Brier Score | **0.028926** |
| Calibrated ECE | **0.001725** |
| Future Balanced Precision | **38.7755%** |
| Future Balanced Recall | **24.6513%** |
| Future Balanced FPR | **1.4035%** |
| Future Intervention Rate | **2.2127%** |
| Locked Future Net Modeled Benefit | **₹49,078.28** |
| 100-Event Benchmark | **100/100 DECIDED** |
| Worker Completion Rate | **~18 transactions/sec** |
| Observed Processing Failures | **0** |
| LightGBM Model Size | **0.41 MB** |

> **Important:** Economic results are modeled evaluation results, not measured production savings.

---

## 1. The Problem

Fraud detection is not only a classification problem.

A payment system must continuously answer:

> **Should this transaction be approved, verified, reviewed, or held?**

Every intervention has a cost.

- Blocking legitimate transactions creates customer friction.
- Verification introduces additional operational/customer cost.
- Manual review consumes investigation capacity.
- Allowing fraudulent transactions creates financial loss.

A useful risk system therefore needs to balance:

```
Fraud Probability
        +
Transaction Exposure
        +
Intervention Cost
        ↓
Operational Decision
```

### Technical challenge

Transaction risk is contextual.

A transaction can appear normal in isolation while becoming suspicious when combined with:

- Customer transaction velocity
- Device activity
- IP activity
- Historical fraud rates
- Payment instrument behavior
- Merchant/product context
- Connected entities

Historical features also introduce a major data leakage risk. Risk Sentinel addresses this with point-in-time historical priors and transaction-time replay.

---

## 2. Our Solution

Risk Sentinel separates fraud detection from business decisioning.

```
Transaction
     │
     ▼
Validation & Feature Extraction
     │
     ├── Velocity Signals
     ├── Historical Fraud Priors
     ├── Entity Relationships
     └── Transaction Attributes
     │
     ▼
Calibrated ML Model
     │
     ▼
Expected Loss
     │
     ▼
Policy Engine
     │
     ├── APPROVE
     ├── VERIFY
     ├── REVIEW
     └── HOLD
     │
     ▼
Audit + Event Publication
```

The system has four core responsibilities:

- **Detect** — Estimate the probability that a transaction is fraudulent.
- **Contextualize** — Combine the prediction with real-time velocity and entity-level context.
- **Quantify** — Translate fraud probability into expected financial loss.
- **Decide** — Use a deterministic policy engine to determine the appropriate action.

---

## 3. Why Risk Sentinel Is Different

Traditional fraud pipelines often look like:

```
Transaction
    ↓
Fraud Probability
    ↓
Threshold
    ↓
Block / Allow
```

Risk Sentinel separates prediction from decisioning.

| Dimension | Conventional Flow | Risk Sentinel |
|---|---|---|
| Primary output | Fraud probability | Probability + operational action |
| Decision | Threshold-based | Policy-driven |
| Financial consequence | Often implicit | Explicit expected loss |
| Context | Transaction-level | Transaction + velocity + entity + historical |
| Calibration | May be absent | Isotonic calibration |
| Historical features | Leakage risk | Point-in-time priors |
| Online state | External | Redis-backed |
| Entity context | Limited | Neo4j graph |
| Failure handling | Application-dependent | Retry + idempotency + DLQ |

### Technical differentiator

Risk Sentinel does not stop at "how likely is fraud?" It asks what intervention is economically justified.

---

## 4. System Architecture

### Components

| Component | Technology | Purpose |
|---|---|---|
| Frontend | Next.js + React + TypeScript | Risk operations dashboard |
| UI | Tailwind / shadcn | Dashboard interface |
| API | FastAPI + Pydantic | Risk and transaction APIs |
| Streaming | Redpanda | Kafka-compatible event streaming |
| Database | PostgreSQL | Durable transaction/risk state |
| Online State | Redis | Velocity and idempotency state |
| Graph | Neo4j | Entity relationship network |
| ML | LightGBM | Fraud-risk prediction |
| Calibration | Isotonic Regression | Probability calibration |
| Explainability | SHAP / reason codes | Risk interpretation |
| Infrastructure | Docker | Reproducible infrastructure |

---

## 5. End-to-End Data Flow

```
1. Transaction received
        ↓
2. Input validation
        ↓
3. Transaction persisted
        ↓
4. Event published / consumed
        ↓
5. Online + historical context assembled
        ↓
6. LightGBM inference
        ↓
7. Probability calibration
        ↓
8. Expected loss calculation
        ↓
9. Policy evaluation
        ↓
10. Risk decision persisted
        ↓
11. Velocity state updated
        ↓
12. Entity graph projected
        ↓
13. Transaction marked DECIDED
        ↓
14. Audit / outbox state retained
```

---

## 6. AI / ML Architecture

### Dataset

Risk Sentinel uses an IEEE-CIS-like public research dataset.

| Dataset | Size |
|---|---|
| train_transaction.csv | ~651.69 MB |
| train_identity.csv | ~25.3 MB |
| test_transaction.csv | ~584.79 MB |
| test_identity.csv | ~24.6 MB |
| **Total** | **~1.29 GB** |

Modeling splits:

| Split | Rows |
|---|---:|
| Training | 413,378 |
| Calibration | 88,581 |
| Future/Test | 88,581 |

The dataset is a research benchmark rather than a production merchant dataset.

### Entity Definitions

```
Customer   = card1
Device     = DeviceInfo
IP         = addr1 + "_" + addr2
Payment    = card1 + "_" + card5 + "_" + card6
Merchant   = ProductCD
```

These are dataset-derived abstractions.

---

## 7. Leakage-Safe Historical Features

V2 introduced historical fraud priors for:

- Customer
- Merchant
- Device
- IP
- Payment instrument

The key constraint is:

```
Historical information
        ↓
TransactionDT < current TransactionDT
```

Replay ordering uses:

- TransactionDT
- TransactionID

Equal-timestamp groups are handled to prevent same-time leakage.

The historical-prior artifact contains:

- 590,540 rows
- 28 columns

Validation included:

- No NaNs
- No duplicate transaction IDs
- 125 spot checks passing

---

## 8. V2 Feature Set

The V2 model combines online behavioral signals with historical fraud priors.

Online features include:

- Transaction amount
- Customer transactions / 1 minute
- Customer transactions / 1 hour
- Device transactions / 1 hour
- IP transactions / 1 hour
- Customer degree
- Device customer count
- IP customer count

V2 extends these with historical entity-level fraud priors.

**Total: 23 model features**

---

## 9. Model Benchmark

Three models were benchmarked on the V2 prior-feature dataset.

| Model | ROC-AUC | PR-AUC | Training Time |
|---|---:|---:|---:|
| CatBoost | 0.829357 | 0.247866 | ~48 sec |
| XGBoost | 0.833574 | 0.265079 | ~135.95 sec |
| LightGBM | 0.840223 | 0.283188 | 12.30 sec |

### LightGBM improvement over CatBoost

**ROC-AUC:** 0.829357 → 0.840223
Absolute improvement: +0.010866
Relative improvement: +1.31%

**PR-AUC:** 0.247866 → 0.283188
Absolute improvement: +0.035322
Relative improvement: +14.25%

LightGBM is therefore the V2 production model.

---

## 10. Model Calibration

Fraud probabilities are used downstream for expected-loss calculation, making probability calibration important. Risk Sentinel uses isotonic regression.

| Metric | Raw LightGBM | Calibrated |
|---|---:|---:|
| ROC-AUC | 0.840223 | 0.840091 |
| PR-AUC | 0.283188 | 0.266300 |
| Brier Score | 0.028849 | 0.028926 |
| ECE | 0.003644 | 0.001725 |

ECE decreased from 0.003644 to 0.001725, while ROC-AUC remained essentially unchanged.

---

## 11. Expected Loss Decisioning

The core economic concept is:

```
Expected Loss = Fraud Probability × Transaction Exposure × Loss Given Fraud
```

Current evaluation assumptions:

| Parameter | Value |
|---|---:|
| LGF | 80% |
| VERIFY cost | ₹20 |
| REVIEW cost | ₹75 |
| HOLD cost | ₹150 |

The policy layer then determines whether intervention is economically justified.

---

## 12. Policy Optimization

Policy thresholds were selected using calibration data and evaluated on a locked future set.

### Calibration-set results

| Policy | Precision | Recall | FPR | Intervention |
|---|---:|---:|---:|---:|
| LOW_FRICTION | 64.92% | 12.23% | 0.2350% | 0.6469% |
| BALANCED | 41.54% | 19.53% | 0.9773% | 1.6143% |
| MAX_RECALL | 41.24% | 19.72% | 0.9995% | 1.6426% |

### Locked future evaluation

| Policy | Precision | Recall | FPR | Intervention |
|---|---:|---:|---:|---:|
| LOW_FRICTION | 62.24% | 12.62% | 0.2760% | 0.7056% |
| BALANCED | 38.78% | 24.65% | 1.4035% | 2.2127% |

> **Important:** The balanced policy had 0.9773% FPR on calibration data, but 1.4035% on the locked future set. The future result is reported exactly and is not represented as a guaranteed ≤1% FPR.

---

## 13. Locked Future Evaluation

The balanced policy was evaluated on:

- Future transactions: 88,581
- Fraud transactions: 3,083
- Fraud prevalence: 3.48%

Confusion matrix:

| | Actual Fraud | Actual Legitimate |
|---|---:|---:|
| Intervention | 760 | 1,200 |
| No Intervention | 2,323 | 84,298 |

Results:

- Precision = 38.7755%
- Recall = 24.6513%
- FPR = 1.4035%

---

## 14. Economic Evaluation

For the balanced policy:

| Metric | Result |
|---|---:|
| Modeled loss avoided | ₹88,278.28 |
| Intervention cost | ₹39,200 |
| Net modeled benefit | ₹49,078.28 |

These are modeled evaluation results, not measured production savings.

---

## 15. LGF Sensitivity

The economic model was evaluated across different Loss Given Fraud assumptions.

| LGF | Loss Avoided | Net Modeled Benefit |
|---|---:|---:|
| 0.60 | ₹66,208.71 | ₹27,008.71 |
| 0.70 | ₹77,243.49 | ₹38,043.49 |
| 0.80 | ₹88,278.28 | ₹49,078.28 |
| 0.90 | ₹99,313.06 | ₹60,113.06 |
| 1.00 | ₹110,347.85 | ₹71,147.85 |

Baseline: LGF = 0.80

---

## 16. Streaming Benchmark

A controlled local benchmark processed 100 valid transactions.

Result:

- 100 / 100 → DECIDED
- 0 observed processing failures

Worker completion rate: ~18 transactions/sec
Producer-side event rate: ~3,199.57 events/sec

*The producer event rate is not worker throughput.*

A separate 200-event benchmark reported an average decision latency of ~97.9 ms:

- p50: 50–100 ms
- p95: 100–250 ms
- p99: 500 ms–1 s

*These are controlled local measurements, not cloud production SLOs.*

---

## 17. Reliability

The streaming worker implements:

### Retry

Up to three attempts.

```
Attempt 1
   ↓
Backoff
   ↓
Attempt 2
   ↓
Backoff
   ↓
Attempt 3
   ↓
DLQ
```

### Idempotency

Repeated events should not create duplicate logical decisions.

### Dead Letter Queue

Events that exhaust retries are routed to the DLQ path.

```
Transaction state

PROCESSING
    │
    ├── Success → DECIDED
    │
    └── Failure → FAILED
```

### Outbox

Database-to-stream publication uses an outbox-oriented design for durable event publication.

---

## 18. Explainability

Risk Sentinel separates risk computation from risk explanation.

The authoritative path is:

```
Probability
     +
Expected Loss
     +
Policy
     ↓
Decision + Reason
```

Example reason code: `ELEVATED_EXPECTED_LOSS`

### LLM boundary

The LLM is not the authoritative fraud decision-maker. LLM functionality is reserved for investigation/explanation workflows. LLM output may assist an investigator, but it must not override the deterministic risk decision.

---

## 19. API

Implemented endpoints include:

```
GET /health
GET /ready

GET /transactions
GET /transactions/{id}
GET /transactions/{id}/risk
GET /transactions/{id}/network

GET /transactions/dashboard
GET /transactions/dashboard/summary
```

FastAPI + Pydantic provide typed API validation. Production authentication and rate limiting remain hardening requirements.

---

## 20. Repository Structure

```
risk-sentinel-v2/
│
├── app/
│   └── frontend/
│
├── backend/
│
├── ml/
│   ├── data/
│   ├── src/
│   │   └── risk_ml/
│   └── artifacts/
│       ├── models/
│       └── reports/
│
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── package.json
└── README.md
```

Large raw datasets are intentionally excluded from Git.

---

## 21. Local Setup

### Prerequisites

- Python 3.13
- Node.js
- Docker
- Docker Compose
- Git
- uv

### Clone

```bash
git clone https://github.com/VISHALKUMAR872/Risk-Manager.git
cd Risk-Manager
```

### Start infrastructure

```bash
docker compose up -d
```

Infrastructure includes:

- PostgreSQL
- Redis
- Neo4j
- Redpanda
- Qdrant

### Backend

```bash
uv sync
```

Start FastAPI using the repository's configured ASGI module. Example:

```bash
uv run uvicorn <application-module>:app --reload --port 8000
```

### Frontend

```bash
npm install
npm run dev
```

Local API: `http://localhost:8000`

Frontend configuration:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 22. Model Artifacts

Production V2 model:
`ml/artifacts/models/fraud_online_v2_priors_lightgbm.txt`

Calibration artifact:
`ml/artifacts/models/fraud_online_v2_priors_lightgbm_isotonic_calibrator.joblib`

Model size: ~0.41 MB

V2 policy reports:

```
ml/artifacts/reports/v2/
├── v2_lightgbm_policy_optimization.json
└── v2_lightgbm_policy_frontier.csv
```

---

## 23. Implemented vs Planned

### Implemented

- FastAPI risk API
- Next.js dashboard
- PostgreSQL persistence
- Redis online state
- Neo4j entity graph
- Redpanda streaming
- Asynchronous risk worker
- Retry handling
- Idempotency
- DLQ handling
- Outbox-oriented publication
- V2 historical fraud priors
- Leakage-safe temporal replay
- CatBoost benchmark
- XGBoost benchmark
- LightGBM benchmark
- Isotonic calibration
- Expected-loss decisioning
- Policy optimization
- Locked future evaluation
- Reliability tests
- Controlled streaming benchmark

### Planned / Hardening

- API authentication
- Production rate limiting
- Explicit PII masking
- Richer SHAP frontend
- Expected-loss alert prioritization
- Policy what-if simulator
- Case-management workflow
- MLflow tracking
- Complete Grafana evidence
- Higher-volume load testing
- Stronger production deployment controls

---

## 24. Limitations

### Dataset

The underlying dataset is a public research benchmark rather than a production merchant dataset.

### Entity semantics

Customer, device, IP, payment and merchant entities are abstractions derived from research-data fields.

### Generalization

Balanced policy FPR:

- Calibration: 0.9773%
- Future: 1.4035%

This demonstrates that threshold performance does not perfectly transfer from calibration data to future data.

### False positives

Balanced future evaluation:

- False positives = 1,200
- Precision = 38.78%

### False negatives

- False negatives = 2,323
- Recall = 24.65%

### Economic assumptions

Results depend on:

- LGF = 80%
- VERIFY = ₹20
- REVIEW = ₹75
- HOLD = ₹150

### Production scale

Current latency and throughput measurements are controlled local benchmarks, not production SLOs.

---

## 25. Security & Production Hardening

Current architecture provides:

- API input validation
- Durable transaction state
- Audit-oriented persistence
- Retries
- Idempotency
- DLQ handling
- Outbox-oriented event publication

Remaining production hardening:

- [ ] API authentication
- [ ] Rate limiting
- [ ] PII masking
- [ ] Production secret management
- [ ] Expanded observability
- [ ] Higher-volume load testing

---

## 26. Scalability

Risk Sentinel uses event-driven processing to decouple ingestion from asynchronous risk computation.

```
API
 │
 ▼
Event Stream
 │
 ├── Worker 1
 ├── Worker 2
 ├── Worker N
 │
 ▼
Risk Decision
```

This creates a natural path toward horizontal worker scaling.

The production LightGBM artifact is approximately 0.41 MB. The current controlled benchmark demonstrates ~18 worker-completed transactions/sec, with 100/100 events → DECIDED.

*These measurements demonstrate current local behavior rather than guaranteed production capacity.*

---

## 27. Observability

The architecture uses:

- OpenTelemetry
- Prometheus
- Grafana

Important operational signals include:

- Transaction latency
- Worker failures
- Retry counts
- DLQ events
- Decision distribution
- Stream health
- Model behavior

*(Grafana dashboard screenshot — to be added before submission.)*

---

## 28. Hackathon Track Alignment

| Criterion | How Risk Sentinel Addresses It |
|---|---|
| Innovation | Economic decisioning rather than probability alone |
| Technical Depth | Streaming + ML + calibration + online state + graph |
| AI/ML Differentiation | Leakage-safe historical priors + model benchmarking |
| Model Quality | 0.840223 ROC-AUC / 0.283188 PR-AUC |
| Calibration | ECE 0.001725 |
| Business Impact | ₹49,078.28 net modeled benefit |
| Scalability | Event-driven architecture + workers |
| Reliability | Retry + idempotency + DLQ + outbox |
| Explainability | Deterministic reason codes + separated explanation layer |
| Production Potential | Durable state + streaming + online features |

---

## 29. Why This Matters

Most fraud systems begin with:

> "Is this transaction fraudulent?"

Risk Sentinel asks the operationally more useful question:

> "Given the probability of fraud, the transaction exposure, the available context, and the cost of intervention, what should we do?"

That distinction drives the architecture.

```
Transaction
     ↓
Real-Time Context
     ↓
ML Inference
     ↓
Calibration
     ↓
Expected Loss
     ↓
Policy
     ↓
Decision
     ↓
Audit / Operations
```

The V2 evaluation provides measurable evidence:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.840223 |
| PR-AUC | 0.283188 |
| Calibrated ECE | 0.001725 |
| Future Precision | 38.7755% |
| Future Recall | 24.6513% |
| Future FPR | 1.4035% |
| Net Modeled Benefit | ₹49,078.28 |

Risk Sentinel does not claim to eliminate fraud. It provides a measurable, explainable and economically-aware framework for deciding how transactions should be handled under fraud risk.
