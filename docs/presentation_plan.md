# RFP Demonstration Strategy: AYS ML Ticket Triage

**Objective**: To demonstrate a high-precision, enterprise-grade ticket triage system that leverages specialized ML models and LLM-augmented training data to solve real-world routing bottlenecks.

---

## 🏛️ Phase 1: The Scenario (The Hook)
*   **The Context**: "Standard enterprise systems rely on keyword mapping. When a complex incident occurs—like a Database Latency issue that looks like an 'Access' problem—traditional systems route it to 'Account Management' instead of 'DBA Ops'. This causes a 4-hour delay (MTTR Drain)."
*   **The Demonstration**: Introduce a **Live "Critical" Ticket** that is semantically complex. 
*   **Key Narrative**: *"We aren't just looking at words; we are looking at behavior and intent."*

---

## 🧠 Phase 2: The Data Genesis (Seed & Augment)
*   **The Problem**: Real-world data is often sparse or messy, making it hard to train high-precision models from scratch.
*   **The Innovation**: **Expert-Seed Augmentation.**
    1.  **Sourcing**: We worked with **Senior Staff Engineers** to curate 200 "Perfect Sample" tickets—real-world edge cases from system logs.
    2.  **LLM Catalyst**: These seeds were passed to the **Gemini LLM** to analyze linguistic patterns and operational context.
    3.  **Synthetic Expansion**: Gemini generated ~5,000 "Contextual Twins"—synthetic tickets that mirror the complexity of the senior staff's data.
*   **Unique Selling Point (USP)**: *"The system is trained on the institutional knowledge of your best engineers, scaled to thousands of training points via Gemini."*

---

## 🛠️ Phase 3: The Engine Room (Precise ML Workflow)
*   **Architecture**: A specialized **Triple-Model Stack** ensuring no single-point-of-failure in logic.
*   **Workflow Mechanicals**:
    1.  **Input Ingestion**: Raw description -> `all-mpnet-base-v2` dense embeddings (Semantic Capture).
    2.  **Layer 1: Severity (RoBERTa)**: A transformer-based model that extracts "Urgency Context" from text. It outputs a severity probability (Critical vs. Routine).
    3.  **Layer 2: Priority (XGBoost)**: A Gradient Boosting machine that combines Layer 1's outputs with real-time metadata (User Status, System Criticality). It calculates the **Business Priority Score**.
    4.  **Layer 3: Routing (Random Forest)**: Maps the 768-dimensional semantic embedding to a specific resolution queue with high precision.
*   **Technicality**: This modular flow allows for independent retraining/tuning of individual layers as business needs shift.

---

## 🛡️ Phase 4: Governance & Trust (The Differentiator)
*   **Challenge**: "Why should we trust the AI to route a P1 incident?"
*   **The Solution**: **The Trust Layer.**
    1.  **Explainability (SHAP)**: We utilize Shapley Values (Game Theory) to provide human-readable **"Insight Pills"** for every prediction. 
        *   *Demo Point*: Show the "Reasoning" in the UI (e.g., *"Keyword 'DNS' + Pattern 'Timeout' influenced Network Team routing"*).
    2.  **The Confidence Gate**: A mathematical safety valve. If the combined model confidence is < 80%, the system flags the ticket for manual review.
*   **Message**: *"This is 'Safe Mode' for Enterprise IT. We prioritize accuracy over blind automation."*

---

## 📉 Phase 5: Impact Analysis (The Commercial Close)
*   **Precise Results**:
    *   **Accuracy**: 92%+ correct first-time routing based on "Seed & Augment" training.
    *   **Velocity**: Routing time reduced from ~15 minutes (manual average) to < 2 seconds.
    *   **Commercial ROI**: ~60% faster Mean Time to Resolution (MTTR) by eliminating routing loops.
*   **Conclusion**: *"We turn institutional knowledge into automated action, ensuring every ticket reaches the right expert, the first time."*

---

> [!IMPORTANT]
> **RFP Cheat Sheet (For Evaluators)**:
> 
> | Question | Direct Technical Answer |
> | :--- | :--- |
> | **How do you handle 'Dirty' data?** | "We use Gemini LLM to clean and expand high-quality 'Expert Seeds' into a robust synthetic training set." |
> | **Is the AI a 'Black Box'?** | "No. We implement SHAP-based explainability to provide a clear audit trail for every automated decision." |
> | **What is the latency?** | "Sub-2 second inference for the entire Triple-Model stack on standard CPU hardware." |
