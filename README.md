# Telecom Customer Churn — Data Quality Audit, EDA & Churn Modeling

An end-to-end pipeline that takes a raw, messy telecom customer file
(`test_datafile.csv`) and turns it into a cleaned dataset, an explained set of
churn drivers, two trained classifiers, and a deployable scoring function.
Everything lives in one executed notebook — **`telecom_data_quality_audit.ipynb`** —
and is regenerable from **`build_notebook.py`**.

The guiding philosophy throughout: **never silently fabricate data, separate
detection from correction, and document every decision with its reasoning and
its measured effect.**

---



## The dataset

| | |
|---|---|
| Rows (raw) | 5,050 |
| Columns | 17 |
| Target | `churned` (0/1) |
| Features | customer demographics, account tenure/contract, billing, usage, support, satisfaction |

---

## Part 1 · Data Quality Audit & Cleaning

We hunted for six classes of defect. For each, we **measured** the problem,
**decided** on an action, and **logged** the number of cells affected so the
before/after summary is computed, not hand-written.

| Step | What we did | **Why** | **How it affected the data** |
|---|---|---|---|
| **Duplicate rows** | Dropped 50 byte-identical rows (kept first) | Identical records carry no new information and would double-weight those customers in every aggregate and model | 5,050 → **5,000 rows** |
| **Inconsistent encodings** | Canonicalized `gender` (M/MALE/male→Male), `internet_service` (fiber→Fiber optic), `phone_service` (Y/N→Yes/No), `payment_method` (CC→Credit card, BT→Bank transfer) | Casing/abbreviation noise fragments a single real category into several, silently splitting groups (`Male` ≠ `male`) and inflating cardinality before any grouping or encoding | **~4,240 cells** re-coded; each categorical collapsed to its true handful of levels. `internet_service="No"` was **kept** as a real value (no internet ≠ missing) |
| **Impossible values** | Converted out-of-domain numbers to `NaN` (e.g. negative `age`/`tenure`/`charges`, `age` > 100) | A `-50` charge or `999` age means the field is *corrupt/unknown*, not that it equals the nearest legal value — so we make it honestly missing rather than clipping (which would invent a plausible-looking number) | **184 cells** → `NaN`; every numeric column now sits inside its physical domain |
| **Sentinel placeholders** | Nulled magic-numbers (`age`=999, `monthly_charges`=9999, `num_support_tickets`=500, `satisfaction`=99) **plus** a statistical sentinel: a 260× spike at `age`=18.00 | Software writes placeholders when a value is unknown. The `age`=18 spike was caught by fractional-part analysis: ages are continuous floats, so a pile-up at exactly 18.00 is a default/floor, not a real cohort | **271 cells** → `NaN` |
| **— the key nuance —** | **Kept** `monthly_charges`=15.00 (109 rows) even though it is also a repeated round number | Prices are categorical-by-plan: many customers legitimately share the *same* plan price, so a repeated round charge is expected. Ages are not drawn from a menu of round numbers. This asymmetry is the heart of *placeholder vs. legitimate-repeat* detection | No change — 109 real customers retained instead of being wrongly nulled |
| **Semantic / cross-field outliers** | Flagged rows where `total_charges` is wildly inconsistent with `monthly_charges × tenure` (ratio outside [0.3, 3]) and nulled only that field | Values can be individually legal but mutually impossible (e.g. a `total_charges` of 218,681 with a normal monthly fee). The ratio clusters tightly around 1.0 (IQR ≈ 0.94–1.07), so a wide band only catches genuine errors, not normal discount/fee variation | **184 cells** of `total_charges` → `NaN` |
| **Missing values** | Produced two artifacts: `df_clean` keeps `NaN`; `df_final` imputes (numeric→**median**, categorical→explicit **"Unknown"**) | Median resists the skew/outliers we just found; we refuse to mode-fill identity fields like gender ("Unknown" is the truthful label). Imputation is a **separate, reversible** final step, not baked into the audit | `df_clean` hands a clean picture of true uncertainty to modeling; `df_final` is a fully-populated convenience copy |

**Net effect:** a de-duplicated, consistently-encoded dataset where every
remaining `NaN` is an *honest* unknown, plus a machine-readable ledger
(`data_quality_audit_log.csv`) of exactly what changed and why.

> Cells flagged by class: encodings 4,240 · missing 1,385 · sentinels 271 ·
> impossible 184 · semantic 184 · duplicate rows 50.

---

## Part 2 · Exploratory Data Analysis

Run on `df_clean` (NaNs preserved) so no imputed values distort relationships.

1. **Overall churn rate = 36.5%** (1,823 / 5,000) — moderately imbalanced, a
   fact that shapes the modeling choices later.

2. **Top 5 churn-associated features — and *why that method*.** A single Pearson
   correlation matrix is **wrong here**: the target is binary and features are
   mixed-type, so Pearson can't handle nominal categories (it would force a fake
   ordering) and captures only linear structure. Instead we used the *right
   metric per type*, both on a shared 0–1 scale, cross-checked with mutual
   information:
   - **numeric → point-biserial |r|** (the correct Pearson specialization for a continuous-vs-binary pair)
   - **categorical → Cramér's V** (χ²-based association on [0,1])

   | Rank | Feature | Metric | Strength | Direction |
   |---|---|---|---|---|
   | 1 | `contract_type` | Cramér's V | 0.291 | — |
   | 2 | `satisfaction_score` | point-biserial | 0.128 | ↓ churn |
   | 3 | `tenure_months` | point-biserial | 0.082 | ↓ churn |
   | 4 | `total_charges` | point-biserial | 0.076 | ↓ churn |
   | 5 | `num_support_tickets` | point-biserial | 0.041 | ↑ churn |

3. **Three visualizations, each titled with its takeaway** (titles generated
   from the computed numbers, so the headline *is* the finding):
   - **Churn by contract type** — month-to-month churns far more than two-year (the dominant lever).
   - **Churn by tenure** — stays high through year 1 (~43%) then drops to ~32% after 24 months → *the first 12 months are the critical retention window*.
   - **Churn by satisfaction** — a steep gradient; the lowest-satisfaction band is the highest-priority intervention group.

4. **Two engineered features (with reasoning, then validated against churn):**
   - **`support_intensity` = tickets / (tenure + 1)** — raw ticket *count* is confounded by tenure (long customers accumulate more); the *rate* isolates real frustration. **Effect:** association with churn rose from 0.041 → **0.076**.
   - **`tenure_group`** (ordinal buckets) — the tenure→churn curve is non-linear, which a single linear term dilutes; buckets let both linear and tree models capture the step-change. **Effect:** churn ≈ 44% in year 1 vs ≈ 31% after, now expressible to the model.

---

## Part 3 · Predictive Modeling

| Choice | What | Why |
|---|---|---|
| Two models | Logistic Regression (linear) + XGBoost (tree-based) | Linear gives a calibrated, interpretable baseline; trees can capture the non-linearities and interactions EDA hinted at |
| Imbalance handling | Class weights, not SMOTE (`class_weight="balanced"`; `scale_pos_weight`) | Churn is only mildly imbalanced (36.5%); class weighting fabricates no rows on a dataset we just cleaned of fake values, and avoids SMOTE's fit-inside-CV leakage hazard |
| Leak-free pipeline | Shared `ColumnTransformer` (median-impute + scale numerics; `"Missing"`-impute + one-hot categoricals), fit on train only | Guarantees serving-time preprocessing matches training exactly |

---

### Hyperparameter Tuning

Both models were tuned using **`RandomizedSearchCV`** (40 iterations each) scored on **ROC-AUC**, with **5-fold `StratifiedKFold`** cross-validation to preserve the class ratio in every fold. The entire preprocessing + estimator pipeline is wrapped inside the search so no data leaks across folds.

| Model | Parameters Searched |
|---|---|
| Logistic Regression | `C` ∈ {0.001 → 100}, `penalty` ∈ {l1, l2}, `solver = saga` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `scale_pos_weight`, `min_child_weight`, `gamma` |

The best parameter combination is automatically refit on the full training set (`refit=True`) before evaluation on the held-out test set.

---

### Results (held-out 25% test set)

| Model | best_cv_roc_auc | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.70 | 0.51 | 0.71 | 0.59 | 0.72 |
| XGBoost | 0.69 | 0.61 | 0.32 | 0.42 | 0.71 |

---

### Why Recall is the Primary Metric

A **false negative** (missed churner) forfeits a customer's entire remaining lifetime value plus re-acquisition cost. A **false positive** (loyal customer flagged) costs only one retention offer. Given this asymmetry:

- **Recall** — the metric to maximize
- **Precision** — a budget guardrail
- **F1** — avoided; it wrongly assumes both error types cost the same
- **ROC-AUC** — used for model selection and threshold calibration

**Recommended workflow:** select the model by ROC-AUC, then lower the decision threshold below 0.50 to push recall to the level the retention budget allows.

---

### Honest Assessment

Absolute scores are modest (AUC ≈ 0.71) and the linear model edges out XGBoost — consistent with Part 2, where only `contract_type` showed strong association with churn. The data itself caps achievable lift; richer behavioral and usage-trend features would be the most impactful next step.

---

## Part 4 · Persistence & Prediction API

- **`churn_model.joblib`** stores the **whole fitted pipeline** (preprocessing +
  best model) — *not* the bare estimator — so inference can never drift from
  training. Bundled metadata (feature lists, §1 normalization maps, tenure-bucket
  definition, risk thresholds) lets the scorer accept raw, dirty input.

- **`predict_churn(customer_data: dict) -> dict`** reloads the artifact from
  disk (proving the round-trip), canonicalizes messy categoricals, recreates the
  engineered features, scores, and returns:

  ```python
  {
    "churn_probability": 0.8384,          # float in [0, 1]
    "risk_tier": "High",                  # Low <0.35 · Medium 0.35–0.60 · High ≥0.60
    "top_risk_factors": [                 # per-customer explanation (coef × this customer's value)
      "contract_type = Month-to-month",
      "satisfaction_score = 2.5",
      "tenure_group = 0-6m"
    ]
  }
  ```

  The explanation is a genuine *local* one (linear contributions for this
  customer), most informative for flagged high-risk customers. *Caveat:*
  `tenure_months` and `tenure_group` are collinear, so for clearly low-risk
  customers the listed factors are weak/near-zero.

---

## Output files

| File | Contents |
|---|---|
| `telecom_data_quality_audit.ipynb` | The full executed notebook (audit → cleaning → EDA → modeling → deployment) |
| `cleaned_data.csv` | Structurally clean data, `NaN`s preserved (recommended for modeling) |
| `cleaned_data_imputed.csv` | + median / `"Unknown"` imputation |
| `cleaned_data_features.csv` | + `support_intensity`, `tenure_group` |
| `data_quality_audit_log.csv` | Machine-readable ledger of every cleaning decision |
| `churn_model.joblib` | Best pipeline (preprocessing + model) + metadata |
| `README.md` | This document |
