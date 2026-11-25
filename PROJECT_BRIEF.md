# Project Brief: Calibrated Stacked Gradient Boosting for 30-Day Readmission Prediction in Diabetic Patients

Author: Vedesh Pandya
Status: data acquired, code not yet written
Target output: IEEE conference paper (~6 pages) + reproducible Kaggle notebook/scripts

---

## 1. Problem Statement

### 1.1 Clinical context
Hospital readmission within 30 days of discharge is a standard quality-of-care indicator. In the US, the Hospital Readmissions Reduction Program (HRRP) financially penalizes hospitals with excess 30-day readmissions. Diabetic patients are a high-risk, high-volume group: diabetes is frequently a secondary diagnosis that complicates the primary reason for admission.

If a hospital can flag high-risk patients at discharge, it can target limited follow-up resources (care-transition calls, early clinic visits, medication reconciliation) at the patients most likely to benefit.

### 1.2 Machine learning formulation
- Unit of prediction: one inpatient encounter (hospital stay) of a diabetic patient.
- Prediction time: at discharge, using only information available during the stay plus prior-utilization counts.
- Task: binary classification.
  - y = 1 if `readmitted == "<30"` (readmitted within 30 days)
  - y = 0 if `readmitted` is `">30"` or `"NO"`
- Output: a calibrated probability of 30-day readmission, plus a binary flag from a threshold chosen on validation data.

### 1.3 Why it is hard
- Class imbalance: only ~11% of encounters are positive (see 2.2).
- Weak signal: the dataset has no vitals, no clinical notes, no post-discharge social data. Modest AUC is expected; the paper must not overclaim.
- Repeated patients: 101,766 encounters come from only 71,518 patients. A random row split leaks patient identity between train and test and inflates results.
- High-cardinality codes: ICD-9 diagnosis codes have hundreds of distinct values.
- Informative missingness: the HbA1c and glucose tests are mostly "not performed", which is itself clinically meaningful (that was the point of the original Strack et al. study).

### 1.4 Success criteria (what makes the paper publishable)
1. Leakage-free evaluation (patient-level split) stated explicitly.
2. Stacked ensemble compared fairly against tuned single models and simple averaging (ablation).
3. Probability quality reported, not just discrimination (Brier score, reliability curves).
4. Imbalance handling and threshold choice done on validation data only.
5. Mean ± std over repeated seeds/folds; no accuracy-only reporting.
6. Interpretability via SHAP.
7. Honest limitations (1999-2008 data, single source, no external validation, not for clinical use).

---

## 2. Dataset

### 2.1 Source
- UCI ML Repository dataset 296: "Diabetes 130-US Hospitals for Years 1999-2008", https://archive.ics.uci.edu/dataset/296, license CC BY 4.0.
- Original study: B. Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records," BioMed Research International, 2014, doi:10.1155/2014/781670.
- Local files (downloaded 2026-10-06, git-ignored):
  - `data/raw/diabetic_data.csv` (SHA-256 `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`)
  - `data/raw/IDS_mapping.csv` (SHA-256 `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e`), the lookup for admission type, discharge disposition and admission source IDs.

### 2.2 Quick facts (from a first sanity check; `01_eda.py` will regenerate these into `results/eda.json`, which is the only source the paper may quote)
- 101,766 encounters x 50 columns; 71,518 unique patients.
- Target distribution: NO 54,864 / >30 35,545 / <30 11,357, so the positive rate is ~11.2%.
- Missing (`?`) fraction: weight ~97%, medical_specialty ~49%, payer_code ~40%, race ~2%, diag_3 ~1.4%.
- 1,652 encounters end in death or hospice (discharge disposition 11, 19, 20, 21). These patients cannot be readmitted.

### 2.3 Parsing gotcha (important)
pandas treats the string `"None"` as NaN by default. In `A1Cresult` and `max_glu_serum`, `"None"` means "test not performed", a real category (~83% and ~95% of rows). Always read with:

```python
pd.read_csv(path, keep_default_na=False, na_values=["?"])
```

### 2.4 Feature groups
- Demographics: race, gender, age (10-year bins).
- Admission context: admission_type_id, admission_source_id, discharge_disposition_id, payer_code, medical_specialty, time_in_hospital.
- Utilization during stay: num_lab_procedures, num_procedures, num_medications, number_diagnoses.
- Prior utilization (previous year): number_outpatient, number_emergency, number_inpatient. Usually the strongest predictors.
- Diagnoses: diag_1 (primary), diag_2, diag_3 as ICD-9 codes.
- Labs: A1Cresult, max_glu_serum.
- Medications: 23 diabetes drugs, each with values No/Steady/Up/Down, plus `change` and `diabetesMed`.
- Identifiers: encounter_id, patient_nbr. Not used as features; patient_nbr is used for grouping the split.

---

## 3. Architecture

### 3.1 High-level pipeline

```
 diabetic_data.csv
        |
 [01_eda.py]  ----------------------------------------> results/eda.json, figures
        |
 [02_preprocess.py]
   - parse ('?' -> NaN, keep 'None' as category)
   - drop: weight, encounter_id (payer_code / specialty: keep with 'Missing' level)
   - exclude expired/hospice discharges (sensitivity analysis with/without)
   - ICD-9 -> clinical groups (circulatory, respiratory, digestive, diabetes,
     injury, musculoskeletal, genitourinary, neoplasms, other)
   - medication summaries (n_meds_changed, n_meds_up, n_meds_down, n_active)
   - utilization total = outpatient + emergency + inpatient
   - patient-level split with StratifiedGroupKFold on patient_nbr
        |
        +--> TRAIN+VAL (80% of patients)            TEST (20% of patients, touched once)
        |
 [03_baselines.py]  tuned LR, RF, XGBoost, LightGBM, CatBoost  -> results/baselines.csv
        |
 [04_stacking.py]
        |
   Level 0 (base learners, 5-fold StratifiedGroupKFold OOF)
   +------------+   +------------+   +------------+
   |  XGBoost   |   |  LightGBM  |   |  CatBoost  |
   +-----+------+   +-----+------+   +-----+------+
         |  OOF p1        |  OOF p2        |  OOF p3
         +-------+--------+-------+--------+
                 |                |
   Level 1   Logistic Regression       Simple average
             meta-learner (on logits)  (ablation baseline)
                 |
   Calibration: isotonic / Platt, fit on held-out OOF predictions
                 |
   Threshold: chosen on validation (max F1 or recall at a fixed precision)
                 |
   Test evaluation: ROC-AUC, PR-AUC, recall, precision, F1, Brier
                 |
 [05_explain.py]  SHAP (TreeExplainer on base models + meta weights) -> figures
```

### 3.2 Why stacking these three
XGBoost, LightGBM and CatBoost are all gradient-boosted trees, but they differ in tree growth (level-wise, leaf-wise, symmetric/oblivious trees) and in categorical handling (one-hot or native, ordered target statistics in CatBoost). Those differences give partially decorrelated errors. A logistic-regression meta-learner is low-variance, hard to overfit on 3 inputs, and its weights are interpretable.

Expectation: stacking same-family models usually gives a small gain over the best single model. The ablation (single vs average vs stack) is a core result whether the gain is large, small or zero.

---

## 4. Methodology (detailed)

### 4.1 Splitting and leakage control
- Outer split: `StratifiedGroupKFold(n_splits=5, groups=patient_nbr, shuffle=True, random_state=42)`. Take one fold as TEST (~20% of patients). Optionally repeat with 5 outer folds to report mean ± std.
- Inner split (within TRAIN): the same grouped/stratified 5-fold for hyperparameter tuning and OOF stack predictions.
- No patient appears in more than one of {train, val, test}. Verify with an assertion and report it in the paper.
- All fitted transforms (encoders, target encoding, imputation, calibration) are fit on training folds only.

### 4.2 Preprocessing decisions
- Drop `weight` (97% missing) and the identifiers.
- `payer_code` and `medical_specialty`: keep, with "Missing" as its own level; collapse rare specialties (< ~1% frequency) to "Other".
- Map ID columns through `IDS_mapping.csv` and merge "unknown / not mapped / NULL" codes.
- ICD-9 grouping following the categories used by Strack et al. [cite]; codes starting with V/E become "Other"; 250.xx becomes "Diabetes".
- Age bins become ordinal integers.
- Medications: ordinal encoding (No=0, Steady=1, Down/Up as separate indicators), plus count features. Drop drugs that are near-constant (e.g. examide, citoglipton have a single value).
- Encoding per model:
  - LR: one-hot + standard scaling.
  - XGBoost / LightGBM: native categorical support (or one-hot for small cardinality).
  - CatBoost: native `cat_features`.

### 4.3 Base learner tuning
- Optuna TPE sampler (seed 42), ~50-100 trials per model, objective = mean inner-CV PR-AUC (better aligned with imbalance than ROC-AUC; report both).
- Early stopping on the inner validation fold.
- Search spaces (indicative):
  - XGBoost: max_depth 3-10, learning_rate 0.01-0.2 (log), subsample 0.6-1, colsample_bytree 0.5-1, min_child_weight 1-20, reg_lambda/alpha (log).
  - LightGBM: num_leaves 15-255, min_child_samples 10-200, feature_fraction, bagging_fraction, lambda_l1/l2.
  - CatBoost: depth 4-10, l2_leaf_reg 1-10, learning_rate, border_count, random_strength.
- Baselines: Logistic Regression (C grid, L2) and Random Forest (RandomizedSearchCV, ~30 iterations).

### 4.4 Out-of-fold stacking
1. For each inner fold k: train each base model on folds != k and predict fold k. Concatenating gives OOF predictions for every training row.
2. Train each base model on all of TRAIN and predict TEST (or average the 5 fold models' test predictions; pick one and state it).
3. Meta-learner: `LogisticRegression` on the logit of the base probabilities, fit on OOF predictions only.
4. Compare with a simple average of the probabilities, and with each single model.

### 4.5 Imbalance handling (ablation)
- A: no handling.
- B: class weights (`scale_pos_weight = n_neg / n_pos` for XGB/LGBM, `auto_class_weights='Balanced'` for CatBoost, `class_weight='balanced'` for LR/RF).
- Report the effect on ranking metrics (often small) and on calibration (weights distort probabilities, which calibration must then fix).
- SMOTE (imbalanced-learn) only if needed; it is generally unnecessary for GBDTs and harms calibration.

### 4.6 Threshold selection
Threshold chosen on validation/OOF predictions only, never on test:
- (i) maximize F1, and
- (ii) the operating point achieving a fixed recall (e.g. 0.70) as a clinically motivated screening point.
Then apply the fixed threshold once to TEST.

### 4.7 Calibration
- Reliability diagram (10 quantile bins) + Brier score, before and after calibration.
- Isotonic vs Platt (sigmoid), fit on OOF predictions with `CalibratedClassifierCV`-style cross-fitting, or a held-out calibration fold.
- Optionally report Expected Calibration Error (ECE) as a secondary number.

### 4.8 Evaluation protocol
- Metrics on TEST: ROC-AUC, PR-AUC (average precision), precision, recall, F1 at the chosen threshold, Brier score.
- Variability: repeat the whole pipeline with 5 outer patient-grouped folds (or 5 seeds) and report mean ± std.
- Statistical comparison: paired bootstrap (1,000 resamples by patient) or DeLong test for stack vs best single model.
- Note the prevalence (~0.11) next to PR-AUC, since the no-skill PR-AUC equals the prevalence.

### 4.9 Explainability
- SHAP TreeExplainer on the best base model (and/or each base model weighted by the meta-learner coefficients).
- Figures: beeswarm summary, top-20 mean |SHAP| bar plot, dependence plots for the top 3 features (likely number_inpatient, discharge disposition, number_diagnoses; verify).

### 4.10 Reproducibility
- `SEED = 42` for numpy, random, sklearn, Optuna, and each booster.
- Pin library versions (Kaggle image versions recorded in `results/env.json`).
- One entrypoint (`run_all.py` or notebook) that executes 01 to 05 in order.
- Every metric goes to `results/*.csv|json`; every figure to `results/*.png` at 300 dpi.

---

## 5. Compute Plan (Kaggle)

### 5.1 Uploading the dataset
A ready-to-upload folder exists at `kaggle_dataset/` (CSV files + `dataset-metadata.json`, id `vedeshp/diabetes-130-us-hospitals-uci-296`). It is created as a private dataset by default:

```bash
kaggle datasets create -p kaggle_dataset
```

In the Kaggle notebook: Add Input, then select your dataset. Files appear under `/kaggle/input/datasets/vedeshp/diabetes-130-us-hospitals-uci-296/`; `code/common.py` locates them automatically. Notebooks are never edited on Kaggle: `tools/kaggle_nb.py` builds them from `code/` and pushes/runs/fetches them (see README).

(Alternative: several public Kaggle copies of this dataset already exist. Uploading our own unmodified copy with checksums is better for reproducibility.)

### 5.2 GPU or CPU
With ~100k rows and ~50-100 features, GBDTs train in seconds to minutes on CPU, so GPU is helpful rather than required. GPU helps most during the Optuna search:
- XGBoost: `device="cuda", tree_method="hist"`
- CatBoost: `task_type="GPU"` (note: GPU CatBoost is not bit-for-bit identical to CPU; fix one and report it)
- LightGBM: the Kaggle GPU build is unreliable, so keep it on CPU (still fast).
Estimated budget: tuning ~1-2 h on a P100/T4, full 5x outer-fold evaluation ~30-60 min. Fits in one Kaggle session (12 h limit).

### 5.3 Repo layout

```
stacked-gbdt-diabetes-readmission/
  code/      01_eda.py 02_preprocess.py 03_baselines.py 04_stacking.py 05_explain.py
  results/   *.csv *.json *.png (single source of truth for the paper)
  paper/     draft.md references.md
  data/raw/  (git-ignored) local copy of the UCI files
  kaggle_dataset/  dataset-metadata.json (+ CSVs, git-ignored)
  PROJECT_BRIEF.md  README.md
```

---

## 6. Novelty: What Can Push Results Beyond a Standard Stack

Plain "stack XGB+LGBM+CatBoost on this dataset" has been done in some form before. The ideas below are ranked by expected impact versus effort. Pick 2-3 as the paper's contributions; each must be evaluated as an ablation (with vs without) under the same patient-level protocol.

### N1. Patient-history (longitudinal) features (highest expected gain)
The dataset contains multiple encounters for many patients, and `encounter_id` increases roughly with time (an assumption to verify and state). For each encounter, compute features from the same patient's earlier encounters only:
- number of prior encounters in the dataset, days-proxy rank of this encounter
- whether any previous encounter was a <30 readmission
- trend of num_medications / number_diagnoses vs the previous stay
- whether the primary diagnosis group changed since the last stay
This uses strictly past information, so it is valid at prediction time. It is compatible with the patient-level split because history is computed within each patient. Most published work treats encounters as independent rows, which makes this the most likely source of real lift.

### N2. Clinically-informed feature engineering
- Comorbidity scoring from the three ICD-9 codes (an approximate Charlson/Elixhauser-style flag set: CHF, renal disease, COPD, etc.) instead of only coarse groups.
- Medication-regimen features: insulin escalation, number of drug classes, any dose change during stay.
- Interaction "HbA1c measured x result x medication changed", which directly extends the Strack et al. finding.

### N3. Representation-diverse stacking
Increase error diversity by giving each base learner a different view of the categorical data: one-hot (XGBoost), native categoricals (LightGBM), ordered target statistics (CatBoost), and optionally a fine-grained ICD-9 target encoding for one extra learner. Diversity is what makes stacking help. Measure pairwise OOF correlation between base learners and report it.

### N4. Feature-augmented (meta-feature) stacking
Instead of the meta-learner seeing only 3 probabilities, also give it a few strong raw features (e.g. number_inpatient, discharge group). This lets it learn where each base model is more reliable, a lightweight form of "feature-weighted linear stacking". Keep it as a logistic regression with interactions to avoid overfitting.

### N5. Calibration-first ensemble selection
Optimize the meta-learner and ensemble weights for log-loss/Brier instead of AUC, then compare isotonic, Platt and beta calibration. Contribution: show the trade-off between discrimination and calibration of stacking under imbalance. This is under-reported in readmission papers.

### N6. Decision-curve analysis (clinical utility)
Report net benefit across threshold probabilities (decision curve analysis) for stack vs single model vs "treat all / treat none". It translates modest AUC gains into whether the model would actually help a care team allocate follow-up. Easy to implement (numpy only), and adds clinical credibility.

### N7. Subgroup and fairness audit
Report AUC, recall and calibration per age group, gender and race subgroup on TEST (with confidence intervals). Flag gaps honestly. Reviewers increasingly expect this for healthcare ML.

### N8. Uncertainty via conformal prediction (optional)
Split-conformal (Mondrian per class) on the calibrated stack gives prediction sets with guaranteed coverage, allowing the model to abstain ("uncertain") on ambiguous patients. A small extra section, but novel for this dataset.

### N9. A neural tabular base learner (GPU use)
Add a GPU-trained tabular deep model (e.g. FT-Transformer or TabM) as a 4th base learner for architectural diversity. This is the main way to make real use of the GPU. Caveat: it needs PyTorch, which is outside the current library list in CLAUDE.md. Treat it as optional and approve explicitly before adding.

### Recommended contribution set for the paper
1. Leakage-free, patient-grouped evaluation with calibrated stacking (core).
2. N1 patient-history features + N2 clinical features (main accuracy lever).
3. N3/N4 diversity-aware stacking with ablation (method novelty).
4. N5 + N6 calibration and decision-curve analysis (clinical relevance).
5. N7 subgroup audit (responsible-AI section, can be short).

---

## 7. Risks and Honest Expectations
- AUC is expected to be modest on this dataset. Stacking may beat the best single model only marginally; report it as is, with confidence intervals.
- encounter_id time ordering (N1) is an assumption; if it cannot be supported, frame the history features as "other encounters observed for the patient" and discuss the limitation, or drop N1.
- Excluding expired/hospice encounters changes the population; report main results with exclusion and a sensitivity analysis without it.
- Data from 1999-2008, US only, single source, no external validation. Not for clinical use.
- This is a retrospective study on public, de-identified data.

---

## 8. Next Steps
1. Upload `kaggle_dataset/` to Kaggle (command in 5.1).
2. Write `code/01_eda.py` and produce `results/eda.json`.
3. `02_preprocess.py` with patient-grouped split + leakage assertion.
4. Baselines, then stacking, calibration, SHAP.
5. Choose the novelty subset (section 6) and run ablations.
6. Draft `paper/draft.md` from `results/` only.
