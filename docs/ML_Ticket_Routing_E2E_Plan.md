# ML-Driven Ticket Routing & Prioritization — End-to-End Implementation Plan

**RFP Reference:** 3.3.2.c — User and Knowledge Support Enhancements (AI/ML Driven Automations)  
**Initiative ID:** 15 — ML-Driven Ticket Routing & Prioritization  
**Domain:** Quality Engineering & AI-Ops  

---

## Executive Summary

This document outlines the complete end-to-end plan for designing, building, validating, and operationalizing an ML-driven ticket routing and prioritization system. The solution replaces manual triage with an intelligent pipeline that automatically assigns priority, severity, and specialist queue — reducing Mean Time to Assign (MTTA) from hours to seconds and eliminating human inconsistency at the point of intake.

---

## Phase Overview

| Phase | Name | Duration | Key Output |
|---|---|---|---|
| 0 | Discovery & Data Audit | 2–3 weeks | Data readiness report |
| 1 | Data Pipeline & Feature Engineering | 3–4 weeks | Clean, labeled training dataset |
| 2 | Model Development & Training | 4–6 weeks | Three trained, validated models |
| 3 | Integration & API Layer | 3–4 weeks | Triage engine deployed behind API |
| 4 | ITSM Integration & UAT | 2–3 weeks | Live auto-population in ITSM |
| 5 | Pilot Rollout & Monitoring | 4 weeks | Production pilot with dashboards |
| 6 | Feedback Loop & Continuous Improvement | Ongoing | Retraining pipeline, model versioning |

**Total estimated timeline:** 18–24 weeks to full production

---

## Phase 0 — Discovery & Data Audit

### Objective
Understand the current state of ticket data, triage workflows, and system integrations before any model work begins.

### Activities

**Stakeholder interviews**
- Interview L1/L2 support leads to understand manual triage logic, common escalation patterns, and known pain points
- Map out all ticket intake channels (portal, email, monitoring, API integrations)
- Document all existing resolver queues and their definitions

**ITSM data audit**
- Export 12–24 months of historical incident tickets from the ITSM platform (ServiceNow / Jira / Remedy)
- Assess data quality: completeness of description fields, consistency of severity/priority labels, resolver queue accuracy
- Identify label noise — tickets where the assigned priority or queue was later corrected by an analyst

**Feasibility assessment**
- Confirm minimum viable data volume (recommend ≥ 5,000 labeled tickets per class)
- Identify class imbalance risks (e.g., very few P1/Critical tickets relative to P4/Low)
- Document any PII or sensitive data in ticket descriptions that may require masking

### Deliverable
Data Readiness Report covering: data volume, label quality score, class distribution, identified gaps, and a go/no-go recommendation.

---

## Phase 1 — Data Pipeline & Feature Engineering

### Objective
Transform raw historical ticket data into a clean, feature-rich dataset suitable for model training.

### Data Collection & Cleaning

Ingest historical tickets with the following fields at minimum:

- `ticket_id` — unique identifier
- `description` — free-text incident description (primary NLP input)
- `subject` / `title` — short-form description
- `created_at` — timestamp
- `source_channel` — portal / email / monitoring / API
- `ci_name` — Configuration Item or affected component
- `affected_service` — business service impacted
- `submitter_id` — user or system that raised the ticket
- `customer_tier` — SLA tier of the submitter (e.g., Enterprise, Standard)
- `severity_label` — ground truth: Critical / High / Medium / Low
- `priority_label` — ground truth: P1 / P2 / P3 / P4
- `assigned_queue` — ground truth: resolver group name
- `resolution_time_hrs` — actual resolution duration (used in feedback loop)
- `was_reassigned` — boolean flag if ticket was moved to a different queue post-assignment

**Cleaning steps:**
1. Remove tickets with blank or very short descriptions (< 10 tokens)
2. Deduplicate near-identical duplicate tickets
3. Strip PII from descriptions (names, emails, IP addresses) using regex + NER masking
4. Normalize queue names (e.g., "Network Ops", "Networking Team", "NW-Ops" → single canonical label)
5. Flag and optionally exclude tickets that were later re-prioritized or reassigned (these indicate labeling noise)

### Feature Engineering

**Text features (for NLP models)**
- Raw tokenized description
- TF-IDF vectors (baseline, interpretable)
- BERT / RoBERTa embeddings via sentence-transformers (semantic, high-performance)
- Extracted keywords: error codes, HTTP status codes, service names, stack trace signals

**Structured features (for tabular models)**
- Hour of day, day of week (time-based urgency signals)
- Customer tier encoded as ordinal (Standard=1, Premium=2, Enterprise=3)
- CI category one-hot encoded
- Source channel one-hot encoded
- Submitter's historical ticket frequency (proxy for power user vs occasional user)
- Severity score passed as input to priority model (sequential dependency)

### Dataset Splits

Partition the cleaned dataset as follows:

- **Training set:** 70% — used to train all three models
- **Validation set:** 15% — used for hyperparameter tuning and early stopping
- **Test set:** 15% — held out, used only for final evaluation; never seen during training

Ensure splits are stratified by class label to preserve class distributions across all three sets.

### Deliverable
Versioned, labeled dataset stored in a feature store or S3/blob bucket. Data pipeline scripts (Python, Apache Airflow or equivalent) documented and reproducible.

---

## Phase 2 — Model Development & Training

### Objective
Train, evaluate, and select the best-performing model for each of the three classification tasks.

---

### Model 1 — Severity Classifier

**Task:** Multi-class classification → Critical / High / Medium / Low

**Recommended approach:** Fine-tuned transformer model

1. Start with a pre-trained base model: `bert-base-uncased` or `roberta-base` from HuggingFace
2. Add a classification head (linear layer over `[CLS]` token)
3. Fine-tune on your labeled incident corpus for 3–5 epochs
4. Apply class-weighted loss function to compensate for class imbalance (Critical tickets are rare)

**Baseline to beat first:** TF-IDF + Logistic Regression (fast to train, strong baseline, interpretable)

**Evaluation metrics:**
- Weighted F1-score (primary metric — accounts for class imbalance)
- Confusion matrix (identify which severity levels are most commonly confused)
- Per-class precision and recall

**Acceptance threshold:** Weighted F1 ≥ 0.82 on the held-out test set

---

### Model 2 — Priority Predictor

**Task:** Multi-class classification → P1 / P2 / P3 / P4

**Recommended approach:** Gradient-boosted decision trees (XGBoost or LightGBM)

Priority is driven primarily by structured business rules (customer tier, SLA, affected user count) combined with the severity signal from Model 1. Gradient-boosted trees excel at this mixed structured-feature problem and are highly interpretable via SHAP values.

**Input features:**
- Predicted severity score from Model 1 (probability distribution, not just the label)
- Customer tier (ordinal encoded)
- Affected service business criticality score (configurable lookup table)
- Time of day / day of week
- Source channel
- Historical escalation rate for this CI/service

**Training procedure:**
1. Train with 5-fold cross-validation on the training set
2. Tune `max_depth`, `learning_rate`, `n_estimators` via Optuna or grid search
3. Generate SHAP feature importance charts for explainability

**Evaluation metrics:**
- Weighted F1-score
- Cohen's Kappa (measures agreement beyond chance — important for ordinal labels like P1–P4)
- SHAP summary plot for stakeholder transparency

**Acceptance threshold:** Weighted F1 ≥ 0.80, Cohen's Kappa ≥ 0.65

---

### Model 3 — Queue Router

**Task:** Multi-class classification → specialist resolver queue (N classes, e.g., Network / Application / Infrastructure / Security / Database / Desktop Support / …)

**Recommended approach:** Fine-tuned text classifier or ensemble of text + structured features

Because queue assignment is heavily driven by the incident description content (keywords, CI references, error type), NLP features dominate. Options:

- **Option A (simpler):** TF-IDF + Random Forest — fast to train, easy to add new queues, strong baseline
- **Option B (higher accuracy):** Fine-tuned BERT with structured feature concatenation in the classification head

For organizations with many queues (> 20), consider a hierarchical classifier: first predict the queue family (Infrastructure, Application, User Support), then predict the specific queue within that family.

**Evaluation metrics:**
- Top-1 accuracy and Top-3 accuracy (is the correct queue in the model's top 3 predictions?)
- Per-queue precision and recall (identify queues with low performance and investigate training data)
- Misrouting rate (how often does the model assign to a completely wrong domain)

**Acceptance threshold:** Top-1 accuracy ≥ 0.78, Top-3 accuracy ≥ 0.93

---

### Confidence Scoring

Each model must output a calibrated confidence score alongside its class prediction (probability of the predicted class). Use temperature scaling or Platt scaling post-training to ensure confidence scores are well-calibrated (i.e., a confidence of 85% should be correct roughly 85% of the time).

**Confidence gate rule:**
- All three model confidence scores ≥ 80% → auto-assign all three fields in ITSM
- Any model confidence < 80% → pre-populate ITSM fields as suggestions, flag for analyst review

### Deliverable
Three trained, versioned model artifacts (`.pkl`, `.pt`, or ONNX format) stored in a model registry (MLflow, SageMaker Model Registry, or Azure ML). Model cards documenting training data, evaluation metrics, known failure modes, and intended use.

---

## Phase 3 — Integration & API Layer

### Objective
Package the three models behind a production-grade inference API that can be called by the ITSM platform at ticket creation time.

### Architecture

```
Ticket Created (ITSM Event)
        │
        ▼
  Triage API Endpoint  ◄──── POST /triage  { ticket_id, description, metadata }
        │
        ├── Feature Extraction Service
        │       └── NLP preprocessing + structured feature assembly
        │
        ├── Model 1: Severity Classifier   ──► severity + confidence
        ├── Model 2: Priority Predictor    ──► priority + confidence
        └── Model 3: Queue Router          ──► queue + confidence
                │
                ▼
         Confidence Gate Logic
                │
        ┌───────┴────────┐
        ▼                ▼
  Auto-assign        Flag for review
  (all ≥ 80%)       (any < 80%)
        │                │
        └───────┬─────────┘
                ▼
   Response payload → ITSM platform
```

### API Contract

**Request:**
```json
POST /api/v1/triage
{
  "ticket_id": "INC0045231",
  "description": "Users in the Mumbai office cannot connect to Outlook. VPN seems functional.",
  "subject": "Email access down - Mumbai",
  "ci_name": "Exchange Online",
  "source_channel": "portal",
  "customer_tier": "enterprise",
  "created_at": "2025-04-06T09:14:00Z"
}
```

**Response:**
```json
{
  "ticket_id": "INC0045231",
  "severity": { "label": "High", "confidence": 0.91 },
  "priority": { "label": "P2", "confidence": 0.87 },
  "queue": { "label": "Network Operations", "confidence": 0.83 },
  "auto_assign": true,
  "flagged_for_review": false,
  "model_version": "v2.4.1",
  "inference_time_ms": 142
}
```

### Non-Functional Requirements

| Requirement | Target |
|---|---|
| API latency (p95) | < 500ms |
| Availability | 99.9% uptime |
| Throughput | ≥ 100 requests/second |
| Fallback behavior | If API unavailable, ITSM routes to manual triage queue |
| Audit logging | Every prediction logged with inputs, outputs, confidence scores, and model version |

### Infrastructure

- Containerize the inference service using Docker
- Deploy on Kubernetes (or equivalent) with horizontal pod autoscaling
- Use a model serving framework (TorchServe, BentoML, or FastAPI + Uvicorn)
- Implement a circuit breaker pattern so ITSM degrades gracefully if the triage API is down

### Deliverable
Deployed triage API in staging environment, passing load tests. API documentation (OpenAPI/Swagger spec). Runbook for on-call engineers.

---

## Phase 4 — ITSM Integration & User Acceptance Testing

### Objective
Connect the triage API to the live ITSM platform and validate end-to-end behavior with real users before production rollout.

### ITSM Integration

Implement a webhook or event listener on ticket creation events in the ITSM platform:

1. Ticket is created → ITSM fires a `ticket.created` webhook
2. Webhook handler calls the Triage API (`POST /api/v1/triage`)
3. API response is used to populate: `severity`, `priority`, `assignment_group` fields
4. If `auto_assign: true` → fields are written directly; SLA timer starts
5. If `flagged_for_review: true` → fields are written as suggestions with a visual indicator (e.g., a tag "ML Suggested — Pending Review"); analyst confirms or overrides

**Override workflow:** When an analyst overrides an ML suggestion, capture:
- Original ML prediction and confidence
- Analyst's correction
- Reason code (optional free-text)

This data feeds the retraining pipeline in Phase 6.

### User Acceptance Testing (UAT)

**Participants:** L1 triage analysts, L2 specialists, support team leads

**UAT scenarios to validate:**
1. New ticket auto-assigned correctly — analyst confirms and resolves
2. Low-confidence ticket flagged — analyst reviews ML suggestion, accepts it
3. Low-confidence ticket flagged — analyst reviews ML suggestion, overrides it
4. ITSM API unavailable — ticket falls to manual queue gracefully
5. Ticket with unusual or edge-case description — observe ML behavior
6. Batch replay: run last 500 historical tickets through the new system and compare ML assignments against the original human assignments

**UAT acceptance criteria:**
- Auto-assignment accuracy ≥ 80% as judged by analyst review panel
- Zero cases where a Critical ticket was auto-assigned as Low
- Analyst satisfaction score ≥ 3.5/5 on ease-of-use survey

### Deliverable
Signed UAT sign-off document. Bug fix log. Go/no-go decision for Phase 5 pilot.

---

## Phase 5 — Pilot Rollout & Monitoring

### Objective
Roll out to a controlled subset of production traffic, monitor performance, and build confidence for full deployment.

### Rollout Strategy

Use a phased traffic split:

- **Week 1–2:** 20% of incoming tickets routed through ML triage; 80% manual
- **Week 3–4:** 50% ML / 50% manual
- **Week 5–6 (if metrics are healthy):** 100% ML with analyst override always available

Use feature flags or a traffic router to control the split without code deployments.

### Monitoring Dashboards

Instrument the following metrics in real time (Grafana, Datadog, or equivalent):

**Model performance metrics:**
- Auto-assignment rate (% of tickets that clear the confidence gate)
- Override rate (% of auto-assignments that analysts subsequently change)
- Per-class accuracy by severity, priority, and queue
- Confidence score distribution over time (watch for drift)

**Operational metrics:**
- API latency (p50, p95, p99)
- Error rate and fallback activation rate
- SLA compliance rate (are tickets being assigned before SLA timer breaches?)

**Business metrics:**
- Mean Time to Assign (MTTA) before vs. after
- Triage analyst hours saved per week
- Misrouting rate (tickets reassigned after initial ML assignment)

### Alerting Rules

| Alert | Threshold | Action |
|---|---|---|
| Override rate spikes | > 25% in any 4-hour window | Page ML Ops team to investigate |
| API latency p95 | > 800ms | Auto-scale inference pods |
| Model confidence avg drops | < 70% for > 1 hour | Trigger manual triage fallback, alert model team |
| Critical ticket misrouted | Any occurrence | Immediate alert, post-mortem required |

### Deliverable
Live monitoring dashboard. Pilot performance report at end of Week 4. Full production go-live decision.

---

## Phase 6 — Feedback Loop & Continuous Improvement

### Objective
Ensure the system gets smarter over time by systematically incorporating analyst corrections, resolution outcomes, and new ticket patterns into model retraining.

### Feedback Data Collection

Every ticket processed by the triage engine generates a feedback record:

```json
{
  "ticket_id": "INC0045231",
  "ml_severity": "High",       "ml_severity_confidence": 0.91,
  "ml_priority": "P2",         "ml_priority_confidence": 0.87,
  "ml_queue": "Network Ops",   "ml_queue_confidence": 0.83,
  "analyst_override": false,
  "final_severity": "High",
  "final_priority": "P2",
  "final_queue": "Network Ops",
  "resolution_time_hrs": 3.2,
  "resolved_by_assigned_team": true,
  "override_reason": null
}
```

These records are stored in a feedback table and aggregated weekly.

### Retraining Cadence

| Trigger | Action |
|---|---|
| Weekly scheduled run | Evaluate model performance on last 7 days of feedback data |
| Override rate > 20% sustained for 3 days | Trigger unscheduled retraining review |
| New resolver queue added | Retrain Queue Router model with new class |
| Model accuracy drops > 5% from baseline | Immediate retraining + regression testing |
| Quarterly | Full retraining on complete historical corpus + new data |

### Retraining Pipeline

1. Pull new feedback-confirmed labels from feedback table
2. Merge with existing training corpus (deduplication, quality check)
3. Retrain all three models (or only affected models if change is isolated)
4. Run automated evaluation suite against held-out test set
5. If new model beats current model on all acceptance thresholds → promote to staging
6. Regression test in staging (shadow mode against live traffic for 48 hours)
7. Blue/green deploy new model version to production

### Model Versioning

- Every model version tagged with: version number, training data date range, evaluation metrics, and changelog
- Rollback procedure documented and tested: revert to previous version in < 5 minutes
- A/B testing capability: route X% of traffic to new model vs. current model for comparison

### Deliverable
Automated retraining pipeline (Airflow or Kubeflow). Model registry with version history. Quarterly model health report template.

---

## Team & Roles

| Role | Responsibilities |
|---|---|
| ML Engineer (×2) | Model development, training pipeline, feature engineering, retraining automation |
| Data Engineer (×1) | Data ingestion, cleaning pipeline, feature store, feedback table |
| Backend Engineer (×1) | Triage API development, containerization, ITSM webhook integration |
| ITSM Admin / Integration Lead (×1) | ITSM-side configuration, UAT coordination, analyst training |
| ML Ops Engineer (×1) | Model serving infrastructure, monitoring dashboards, alerting, CI/CD for models |
| QA Engineer (×1) | UAT test case design, regression testing, performance testing |
| Project Manager (×1) | Timeline, stakeholder communication, risk tracking |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Insufficient historical ticket data | Medium | High | Supplement with synthetic data; lower confidence thresholds initially |
| High label noise in historical data | High | High | Manual label review of sample; train on analyst-confirmed subset first |
| ITSM API limitations blocking integration | Low | High | Early integration spike in Phase 3; escalate vendor support |
| Analyst resistance to ML suggestions | Medium | Medium | Transparent confidence display; override always available; UAT buy-in |
| Model drift over time (accuracy degrades) | High | Medium | Automated monitoring + retraining pipeline from Phase 6 |
| Critical tickets misclassified | Low | Critical | Hard-coded rule: any ticket mentioning predefined critical CIs bypasses ML and gets P1/Critical automatically |

---

## Success Metrics

| KPI | Baseline (current) | Target (6 months post-launch) |
|---|---|---|
| Mean Time to Assign (MTTA) | > 2 hours | < 2 minutes |
| Triage accuracy (vs. analyst ground truth) | N/A (manual) | ≥ 85% |
| Auto-assignment rate | 0% | ≥ 70% of all tickets |
| Override rate | N/A | ≤ 15% |
| SLA breach rate due to late assignment | Baseline TBD | Reduce by 40% |
| Analyst triage hours per week | Baseline TBD | Reduce by 60% |

---

## Appendix A — Technology Stack (Reference)

| Layer | Recommended Tools |
|---|---|
| Data processing | Python, Pandas, Apache Spark (for large volumes) |
| NLP / Embeddings | HuggingFace Transformers, sentence-transformers |
| ML training (tabular) | XGBoost, LightGBM, scikit-learn |
| Experiment tracking | MLflow |
| Model serving | FastAPI + Uvicorn, BentoML, or TorchServe |
| Infrastructure | Docker, Kubernetes, Helm |
| Orchestration | Apache Airflow or Kubeflow Pipelines |
| Monitoring | Grafana + Prometheus, or Datadog |
| ITSM platforms | ServiceNow (REST API), Jira Service Management (API), BMC Remedy |

---

## Appendix B — Glossary

| Term | Definition |
|---|---|
| MTTA | Mean Time to Assign — the average time between ticket creation and assignment to a resolver |
| CI | Configuration Item — any component managed by ITSM (server, application, network device) |
| Confidence gate | The logic that decides whether to auto-assign or flag for human review based on model confidence scores |
| Fine-tuning | Taking a pre-trained ML model and continuing its training on a domain-specific dataset |
| SHAP | SHapley Additive exPlanations — a method for explaining individual ML model predictions |
| Drift | The phenomenon where a model's real-world accuracy degrades over time as data patterns shift |
| Label noise | Incorrect ground-truth labels in training data caused by inconsistent human annotation |

---

*Document version: 1.0 | Prepared for RFP Section 3.3.2.c | Classification: Internal / RFP Use*
