# ML-Driven Ticket Routing & Prioritization — Implementation Plan

**Project:** VZ Incident Triage Automation  
**RFP Reference:** Section 3.3.2.c — AI/ML Driven Automations  
**Data Mode:** Synthetic training data (RFP demo)  
**Priority Scale:** P1 (highest) → P5 (lowest)  
**Ticket Format:** `VZ_INCxxxxx`

---

## Locked Schema

### Inference-time inputs (what arrives per ticket)

```
ticket_no      → VZ_INC01234         (identifier only — not a model feature)
created_at     → 06-04-2026 13:48:02  (DD-MM-YYYY HH:MM:SS)
affected_user  → Wolford, Travis      (Last, First)
description    → "OS356034 is stuck in Approved in BGW..." (primary NLP input)
```

### ML outputs (what the system predicts)

```
severity    → Critical | High | Medium | Low          (Model 1 — RoBERTa)
priority    → P1 | P2 | P3 | P4 | P5                 (Model 2 — XGBoost)
assigned_to → BGW Team | Network Ops | App Support…   (Model 3 — Random Forest)
```

### Training-only fields (historical ground truth)

```
severity        → label for Model 1
priority        → label for Model 2
assigned_to     → label for Model 3
escalation_count → feature for Model 2 + Model 3
reassign_count  → feature for Model 3
reopen_count    → feature for Model 2
state           → filter gate — use only "resolved" tickets for training
resolved_at     → used to compute resolution_time_mins (derived feature)
```

### Dropped fields (not used)

```
resolution_notes  → team's internal notes, out of ML scope
link(ticket_id)   → deduplication only, not a model feature
ticket_id         → internal system ID, identifier only
```

---

## Phase 1 — Synthetic Data Generation

**File:** `data/generate_synthetic_data.py`  
**Output:** `data/vz_tickets_synthetic.csv` — 1,000 labeled tickets

### Ticket pool design

Generate realistic VZ-style descriptions across 5 named teams:

| Team (assigned_to) | Issue Domain | Example Description Patterns |
|---|---|---|
| `BGW Team` | Order flow, BGW approvals, PC sync | "OS{id} stuck in Approved in BGW", "order did not flow to PC", "children not generated in BGW" |
| `Network Ops` | Connectivity, VPN, routing | "users cannot connect to VPN", "site-to-site tunnel down", "packet loss on MPLS link" |
| `App Support` | Application errors, login failures | "login page returns 500 error", "app crashes on submit", "session timeout after 2 minutes" |
| `Infrastructure` | Server, storage, CPU, memory | "prod server CPU at 99%", "disk full on node 3", "VM unresponsive after reboot" |
| `Desktop Support` | Laptop, printer, peripherals | "laptop won't boot after update", "printer offline", "USB docking station not detected" |

### Severity → Priority correlation rules

These rules ensure synthetic data reflects realistic business logic:

```
Critical  → always P1 or P2
High      → P2 or P3
Medium    → P3 or P4
Low       → P4 or P5

escalation_count > 0  → severity bumps up one level
reopen_count > 1      → priority bumps up one level
```

### Synthetic field generation rules

```
ticket_no       → VZ_INC + zero-padded sequential int (VZ_INC00001 … VZ_INC01000)
created_at      → random datetime, business hours weighted (08:00–18:00 IST, Mon–Fri)
affected_user   → random full name pool (Last, First format)
escalation_count → 70% → 0,  20% → 1,  10% → 2+
reassign_count  → 75% → 0,  20% → 1,   5% → 2+
reopen_count    → 80% → 0,  15% → 1,   5% → 2+
state           → all set to "resolved" (training data only)
resolved_at     → created_at + random delta (10 mins to 48 hrs, severity-weighted)
```

### Class distribution targets (1,000 tickets)

```
Severity:  Critical=80,  High=280,  Medium=400,  Low=240
Priority:  P1=60,  P2=160,  P3=380,  P4=270,  P5=130
Teams:     BGW=220, NetworkOps=200, AppSupport=220, Infra=180, Desktop=180
```

---

## Phase 2 — Feature Engineering

**File:** `data/feature_engineering.py`

### From `description` (text features)

Processed by RoBERTa tokenizer for Model 1 and sentence-transformers for Model 3.

```
raw_text         → cleaned description (lowercase, strip special chars)
token_count      → number of words (proxy for issue complexity)
has_error_code   → bool — regex match for patterns like OS\d+, ERR_, HTTP 5xx
system_keywords  → extracted system names: BGW, PC, VPN, MPLS, SAP, etc.
bert_embedding   → 768-dim vector from sentence-transformers (for Random Forest)
```

### From `created_at` (temporal features)

```
hour_of_day      → int 0–23
day_of_week      → int 0–6 (0=Monday)
is_business_hour → bool (08:00–18:00 IST, Mon–Fri)
is_weekend       → bool
```

### From `affected_user` (user features)

```
user_ticket_frequency → count of past tickets by this user (from training set)
user_avg_severity     → historical average severity for this user
```

### From count fields (operational features)

```
escalation_count → int (as-is)
reassign_count   → int (as-is)
reopen_count     → int (as-is)
resolution_time_mins → (resolved_at - created_at) in minutes — training only
```

### Train / Validation / Test split

```
Training:   70%  → 700 tickets  (model fitting)
Validation: 15%  → 150 tickets  (hyperparameter tuning)
Test:       15%  → 150 tickets  (final evaluation — never seen during training)

Split method: stratified by severity label to preserve class distribution
```

---

## Phase 3 — Model 1: Severity Classifier (RoBERTa)

**File:** `models/train_severity.py`  
**Input:** `description` (free text)  
**Output:** `severity` — Critical | High | Medium | Low  
**Why RoBERTa:** Severity is a language problem. "OS356034 stuck in Approved" vs "OS356034 did not flow to PC and children missing" carry different urgency that only a transformer understands. TF-IDF sees similar tokens; RoBERTa reads consequence.

### Model architecture

```
Base:           roberta-base (HuggingFace)
Head:           Linear(768 → 4)  — 4-class classification
Tokenizer:      RobertaTokenizer, max_length=256, truncation=True
Loss:           CrossEntropyLoss with class weights (handle Critical imbalance)
```

### Training config

```python
BASE_MODEL    = "roberta-base"
MAX_LEN       = 256
BATCH_SIZE    = 16
EPOCHS        = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
METRIC        = "weighted_f1"   # primary metric
```

### Class weight formula

```python
# Inverse frequency weighting — upweights rare Critical class
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=[0,1,2,3], y=y_train)
```

### Label encoding

```
0 → Critical
1 → High
2 → Medium
3 → Low
```

### Evaluation targets

```
Weighted F1       ≥ 0.82   (primary — accounts for class imbalance)
Critical Recall   ≥ 0.90   (must not miss Critical tickets)
Confusion matrix  — reviewed manually for Critical↔High misclassification
```

### Output format

```python
{
  "label": "High",
  "probabilities": {
    "Critical": 0.05,
    "High":     0.91,
    "Medium":   0.03,
    "Low":      0.01
  },
  "confidence": 0.91   # max probability
}
```

### Artifacts saved

```
models/severity_model/         → full RoBERTa model directory
models/severity_tokenizer/     → tokenizer
models/severity_label_map.json → {0: "Critical", 1: "High", ...}
```

---

## Phase 4 — Model 2: Priority Predictor (XGBoost)

**File:** `models/train_priority.py`  
**Input:** structured features + severity probability vector from Model 1  
**Output:** `priority` — P1 | P2 | P3 | P4 | P5  
**Why XGBoost:** Priority is a business rules problem, not a language problem. The same description from an enterprise user at 02:00 with escalation_count=2 is P1; from a standard user at 14:00 with no escalation it is P3. XGBoost handles these non-linear tabular interactions and produces SHAP values for RFP audit trails.

### Feature vector (per ticket)

```python
features = [
    severity_prob_Critical,   # float — from Model 1 output
    severity_prob_High,       # float
    severity_prob_Medium,     # float
    severity_prob_Low,        # float
    hour_of_day,              # int 0–23
    day_of_week,              # int 0–6
    is_business_hour,         # bool → int
    is_weekend,               # bool → int
    escalation_count,         # int
    reopen_count,             # int
    token_count,              # int — description length proxy
    user_ticket_frequency,    # int
    user_avg_severity_encoded # int 0–3
]
# Total: 13 features
```

### Training config

```python
MODEL = xgb.XGBClassifier(
    n_estimators      = 400,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    eval_metric       = "mlogloss",
    early_stopping_rounds = 30,
    random_state      = 42
)
# 5-fold stratified cross-validation on training set
# Hyperparameter tuning via Optuna (20 trials)
```

### Label encoding

```
0 → P1   (highest urgency)
1 → P2
2 → P3
3 → P4
4 → P5   (lowest urgency)
```

### SHAP explanation (per prediction)

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_input)

# Example output for INC01234:
# escalation_count=1   → +0.42 toward P1
# severity_Critical    → +0.38 toward P1
# is_business_hour=1   → -0.21 toward P2 (staff available)
# hour_of_day=13       → -0.18 toward P2
```

### Evaluation targets

```
Weighted F1     ≥ 0.80
Cohen's Kappa   ≥ 0.65   (ordinal agreement — P1 vs P2 confusion is less bad than P1 vs P5)
P1 Precision    ≥ 0.88   (must not over-escalate)
P1 Recall       ≥ 0.90   (must not miss P1s)
```

### Artifacts saved

```
models/priority_model.pkl       → trained XGBoost model
models/priority_shap_explainer  → SHAP TreeExplainer
models/priority_feature_names.json
```

---

## Phase 5 — Model 3: Queue Router (Random Forest)

**File:** `models/train_queue.py`  
**Input:** BERT embedding (768-dim) + structured metadata  
**Output:** `assigned_to` — BGW Team | Network Ops | App Support | Infrastructure | Desktop Support  
**Why Random Forest:** Queue routing is a mixed-signal problem — part language (what kind of issue?), part metadata (which system, which user?). Random Forest handles 768-dimensional embeddings efficiently, produces reliable per-class probabilities for the confidence gate, and runs inference in under 5ms. Naive Bayes on keywords alone would have misrouted the sample ticket — "BGW" and "PC" in the description is the primary signal, not generic system keywords.

### Feature vector (per ticket)

```python
features = np.concatenate([
    bert_embedding,           # 768-dim float array from sentence-transformers
    [
        hour_of_day,          # int
        is_business_hour,     # int
        escalation_count,     # int
        reassign_count,       # int
        token_count,          # int
        has_error_code,       # int 0/1
        user_ticket_frequency # int
    ]
])
# Total: 775 features (768 embedding + 7 structured)
```

### Sentence-transformer model

```python
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-mpnet-base-v2")
# 768-dim, semantic quality, strong on technical domain text
# Batch encode: embedder.encode(descriptions, batch_size=64, normalize_embeddings=True)
```

### Training config

```python
MODEL = RandomForestClassifier(
    n_estimators  = 500,
    max_depth     = None,
    min_samples_leaf = 2,
    class_weight  = "balanced",
    n_jobs        = -1,
    random_state  = 42
)
```

### Label encoding

```
0 → BGW Team
1 → Network Ops
2 → App Support
3 → Infrastructure
4 → Desktop Support
```

### Output format

```python
{
  "label": "BGW Team",
  "probabilities": {
    "BGW Team":       0.87,
    "App Support":    0.08,
    "Network Ops":    0.03,
    "Infrastructure": 0.02,
    "Desktop Support":0.00
  },
  "confidence": 0.87,
  "top_3": ["BGW Team", "App Support", "Network Ops"]
}
```

### Evaluation targets

```
Top-1 Accuracy  ≥ 0.78
Top-3 Accuracy  ≥ 0.93
Per-team F1     reviewed — Desktop Support is smallest class, monitor closely
```

### Artifacts saved

```
models/queue_model.pkl          → trained Random Forest
models/queue_embedder/          → sentence-transformer model
models/queue_label_encoder.pkl  → LabelEncoder
```

---

## Phase 6 — Confidence Gate Logic

**File:** `api/confidence_gate.py`

All three model outputs carry a confidence score. The gate decides auto-assign vs flag for review.

```python
THRESHOLD = 0.80

def evaluate_gate(sev_conf, pri_conf, queue_conf):
    all_clear = all(c >= THRESHOLD for c in [sev_conf, pri_conf, queue_conf])
    low_fields = [
        field for field, conf in [
            ("severity", sev_conf),
            ("priority", pri_conf),
            ("assigned_to", queue_conf)
        ] if conf < THRESHOLD
    ]
    return {
        "auto_assign":        all_clear,
        "flagged_for_review": not all_clear,
        "low_confidence_fields": low_fields
    }
```

### Gate outcomes

```
All 3 scores ≥ 0.80  → auto_assign: true   → ITSM fields written directly
Any score   < 0.80   → flagged_for_review   → fields shown as ML suggestions
                                              → analyst confirms or overrides
```

---

## Phase 7 — FastAPI Inference Service

**File:** `api/main.py`

### Endpoint

```
POST /api/v1/triage
```

### Request schema

```python
class TriageRequest(BaseModel):
    ticket_no:     str        # VZ_INC01234
    created_at:    str        # "06-04-2026 13:48:02"  DD-MM-YYYY HH:MM:SS
    affected_user: str        # "Wolford, Travis"
    description:   str        # full issue text
```

### Response schema

```python
class TriageResponse(BaseModel):
    ticket_no:    str
    severity:     dict   # {"label": "High", "confidence": 0.91}
    priority:     dict   # {"label": "P2",   "confidence": 0.87}
    assigned_to:  dict   # {"label": "BGW Team", "confidence": 0.87, "top_3": [...]}
    shap_reasons: list   # ["escalation_count pushed toward P1", ...]
    auto_assign:  bool
    flagged_for_review: bool
    low_confidence_fields: list
    model_version:     str    # "v1.0.0"
    inference_time_ms: int
```

### Inference pipeline (per request)

```
1. Parse created_at → extract hour_of_day, day_of_week, is_business_hour, is_weekend
2. Compute user_ticket_frequency from in-memory user history cache
3. Tokenize description → RoBERTa → severity label + probability vector
4. Build 13-feature vector → XGBoost → priority label + confidence + SHAP values
5. Encode description → sentence-transformer → 775-feature vector → Random Forest → queue label + probabilities
6. Run confidence gate on three confidence scores
7. Return unified TriageResponse
```

### Non-functional targets

```
p95 latency    < 500ms
Availability   99.9%
Fallback       if API down → ITSM routes to manual queue (no data loss)
Audit log      every prediction stored: inputs + outputs + confidence + model_version
```

---

## Phase 8 — Demo UI

**File:** `demo/index.html` — single-file, no build step

### UI screens

```
Screen 1 — Ticket input form
  Fields:  ticket_no, created_at, affected_user, description (textarea)
  Pre-fills: sample ticket VZ_INC01234 data on load
  Action:  "Run ML Triage" button → calls FastAPI

Screen 2 — Results panel (same page, revealed after call)
  Shows:
    - Severity badge (color-coded: Critical=red, High=orange, Medium=yellow, Low=green)
    - Priority badge (P1–P5)
    - Assigned team badge
    - Confidence bar for each model
    - SHAP reasons (why this priority?)
    - auto_assign green banner OR flagged_for_review amber banner
    - Inference time

Screen 3 — Model comparison tab
  Side-by-side:  RoBERTa vs TF-IDF score comparison for current description
  Shows why BERT wins on this ticket type

Screen 4 — Audit log table
  All predictions made this session
  Columns: ticket_no, severity, priority, assigned_to, confidence, auto_assign, time
```

---

## Phase 9 — Feedback Loop (Post-Demo / Production)

**File:** `api/feedback.py`

```python
# Called when analyst overrides an ML prediction in ITSM
POST /api/v1/feedback
{
  "ticket_no":      "VZ_INC01234",
  "field":          "priority",         # which field was wrong
  "ml_value":       "P2",
  "correct_value":  "P1",
  "analyst_id":     "analyst_007"
}
# Stored in feedback.jsonl for next retraining cycle
```

### Retraining triggers

```
Weekly scheduled    → merge feedback + retrain all 3 models
Override rate > 20% for 3 days → immediate unscheduled retrain
New team added      → retrain Model 3 only
Model accuracy drops > 5% from baseline → emergency retrain
```

---

## File Structure

```
ml_triage/
├── data/
│   ├── generate_synthetic_data.py   # Phase 1 — 1,000 VZ-style tickets
│   ├── feature_engineering.py       # Phase 2 — all feature transforms
│   └── vz_tickets_synthetic.csv     # generated output
│
├── models/
│   ├── train_severity.py            # Phase 3 — RoBERTa fine-tuning
│   ├── train_priority.py            # Phase 4 — XGBoost + SHAP
│   ├── train_queue.py               # Phase 5 — Random Forest + embeddings
│   ├── severity_model/              # saved RoBERTa model
│   ├── severity_tokenizer/          # saved tokenizer
│   ├── priority_model.pkl           # saved XGBoost
│   ├── priority_shap_explainer      # saved SHAP explainer
│   ├── queue_model.pkl              # saved Random Forest
│   ├── queue_embedder/              # saved sentence-transformer
│   └── queue_label_encoder.pkl      # saved LabelEncoder
│
├── api/
│   ├── main.py                      # Phase 7 — FastAPI service
│   ├── confidence_gate.py           # Phase 6 — gate logic
│   ├── feedback.py                  # Phase 9 — override capture
│   └── schemas.py                   # Pydantic request/response models
│
├── demo/
│   └── index.html                   # Phase 8 — single-file demo UI
│
├── requirements.txt
└── README.md
```

---

## Requirements

```
# requirements.txt
transformers==4.40.0
sentence-transformers==3.0.0
torch==2.3.0
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.45.0
fastapi==0.111.0
uvicorn[standard]==0.29.0
pandas==2.2.2
numpy==1.26.4
faker==25.0.0          # synthetic user name generation
python-dateutil==2.9.0
pydantic==2.7.1
```

---

## Build Order

```
Step 1  →  generate_synthetic_data.py     produces vz_tickets_synthetic.csv
Step 2  →  feature_engineering.py         produces feature matrices for all 3 models
Step 3  →  train_severity.py              produces RoBERTa model artifacts
Step 4  →  train_priority.py              produces XGBoost + SHAP artifacts
Step 5  →  train_queue.py                 produces Random Forest + embedder artifacts
Step 6  →  api/main.py                    FastAPI service loading all 3 models
Step 7  →  demo/index.html                UI calling the FastAPI endpoint
Step 8  →  test end-to-end with VZ_INC01234 sample ticket
```

---

## Success Criteria

| Model | Metric | Target |
|---|---|---|
| RoBERTa — Severity | Weighted F1 | ≥ 0.82 |
| RoBERTa — Severity | Critical Recall | ≥ 0.90 |
| XGBoost — Priority | Weighted F1 | ≥ 0.80 |
| XGBoost — Priority | Cohen's Kappa | ≥ 0.65 |
| XGBoost — Priority | P1 Recall | ≥ 0.90 |
| Random Forest — Queue | Top-1 Accuracy | ≥ 0.78 |
| Random Forest — Queue | Top-3 Accuracy | ≥ 0.93 |
| FastAPI | p95 Latency | < 500ms |
| Confidence Gate | Auto-assign rate | ≥ 70% of tickets |

---

*Plan version 1.0 — ready for implementation. Build order: Step 1 → Step 8 sequentially.*
