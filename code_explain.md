# Code Explained

This document walks through every file and every function in the repository: what it does, why it is written that way, what goes in and what comes out. Read it top to bottom once; afterwards use it as a reference.

Contents

1. The big picture
2. How a run flows (data flow between files)
3. Run configuration (environment variables)
4. `code/common.py` — shared settings and helpers
5. `code/features.py` — cleaning, feature construction, preprocessors
6. `code/models.py` — model pipelines, tuning, the stacking estimator
7. `code/evaluation.py` — metrics, thresholds, calibration, bootstrap, plots
8. `code/01_eda.py` — exploratory data analysis
9. `code/02_preprocess.py` — features and the patient-level split
10. `code/03_baselines.py` — tuning and single-model evaluation
11. `code/04_stacking.py` — stacking, averaging, calibration, thresholds
12. `code/05_explain.py` — SHAP explanations
13. `tools/kaggle_nb.py` — building and running Kaggle notebooks
14. Other files (`tools/kaggle_config.json`, `kaggle_dataset/`, `requirements.txt`, `.gitignore`)
15. Every output file and which script writes it
16. Known limitations and things to watch
17. How to change common things

---

## 1. The big picture

The task: for each hospital stay ("encounter") of a diabetic patient, predict the probability that the patient is readmitted within 30 days.

The code is split in two kinds of files:

- **Numbered scripts** (`01_eda.py` … `05_explain.py`) are the steps of the experiment. Each one is run as `python code/0X_name.py`, reads its inputs from disk, and writes its outputs to `results/` (numbers and figures) or `data/processed/` (large intermediate files).
- **Helper modules** (`common.py`, `features.py`, `models.py`, `evaluation.py`) contain the reusable functions. They are imported by the scripts and never run on their own.

Three design rules run through the whole code base:

1. **No leakage.** Anything that learns from data (encoders, scalers, models, calibrators, thresholds) is fitted only on training data, and all cross-validation folds are grouped by patient so the same person never appears on both sides of a split.
2. **One source of truth.** Every number in the paper comes from a file in `results/`.
3. **Reproducible.** The random seed is 42 everywhere, and the same code runs locally and on Kaggle without changes.

## 2. How a run flows

```
data/raw/diabetic_data.csv  (or /kaggle/input/.../diabetic_data.csv)
        │
        ├── 01_eda.py ──────────────► results/eda*.json|csv|png
        │
        └── 02_preprocess.py ───────► data/processed/dataset.joblib   (X, y, groups, is_test, ...)
                                      results/split.json, results/features.csv
                │
                ├── 03_baselines.py ► results/best_params.json, tuning_trials_*.csv,
                │                     baselines_cv.csv, baselines_test.csv, imbalance_ablation.csv
                │                     data/processed/preds_baselines.joblib
                │
                └── 04_stacking.py ─► (needs results/best_params.json from 03)
                                      results/stacking_*.csv, calibration_comparison.csv, ...
                                      data/processed/stacker_seed42.joblib
                        │
                        └── 05_explain.py ► (needs stacker_seed42.joblib and stacking_comparison.csv)
                                            results/shap_*.csv|png, shap_additivity_check.json
```

Dependencies: 01 and 02 only need the raw CSV. 03 needs 02. 04 needs 02 and 03. 05 needs 02 and 04.

## 3. Run configuration (environment variables)

All settings live at the top of `common.py` and can be overridden without editing code, by setting an environment variable before running a script (locally) or with `--env` (on Kaggle).

| Variable | Default | Meaning |
|---|---|---|
| `FAST` | `0` | `1` = smoke test: 15% of patients, tiny budgets, outputs go to `results_fast/` and `data/processed_fast/` so real results are never overwritten |
| `N_TRIALS` | 40 (FAST: 3) | Maximum Optuna trials per model |
| `TUNE_TIMEOUT` | 900 (FAST: 60) | Maximum tuning time per model, seconds |
| `TUNE_FOLDS` | 3 | How many of the 5 grouped folds are used while tuning (speed) |
| `CV_FOLDS` | 5 | Number of grouped folds for out-of-fold predictions |
| `N_SEEDS` | 3 (FAST: 1) | Repetitions of the stacking experiment (seeds 42, 43, 44, …) |
| `N_BOOT` | 1000 (FAST: 50) | Bootstrap resamples for confidence intervals |
| `N_JOBS` | -1 | CPU threads for the models (-1 = all) |
| `USE_GPU` | auto | `1`/`0` forces XGBoost onto GPU/CPU; otherwise detected with `nvidia-smi` |
| `READMISSION_DATA_DIR` | auto | Folder containing `diabetic_data.csv`, if not in the default places |
| `EXCLUDE_EXPIRED` | `1` | `0` keeps death/hospice encounters (for the sensitivity analysis) — read by `02_preprocess.py` |

---

## 4. `code/common.py`

Shared constants and small helpers used by every script.

### Constants

- `SEED = 42` — the one random seed used everywhere.
- `FAST`, `N_TRIALS`, `TUNE_TIMEOUT`, `TUNE_FOLDS`, `CV_FOLDS`, `N_SEEDS`, `N_BOOT`, `N_JOBS` — see Section 3.
- `FAST_N_ROWS = 15000` — approximate size of the FAST-mode subsample.
- `SEEDS = [42, 43, 44, …]` — `N_SEEDS` consecutive seeds starting at 42, used by `04_stacking.py`.
- `REPO_ROOT` — the folder above `code/`, found from this file's own location. On Kaggle this becomes `/kaggle/working`, because the notebook writes the code files to `/kaggle/working/code/`. That is why all paths work in both places without changes.
- `RESULTS_DIR` — `results/` (or `results_fast/` in FAST mode).
- `PROCESSED_DIR` — `data/processed/` (or `data/processed_fast/`).
- `DATASET_FILE`, `STACKER_FILE` — paths of the two large intermediate files.
- `RAW_FILE = "diabetic_data.csv"`, `IDS_FILE = "IDS_mapping.csv"` — raw file names.
- `TARGET_RAW = "readmitted"` (original 3-class column) and `TARGET = "readmit_30d"` (our 0/1 label).
- `EXPIRED_IDS = [11, 19, 20, 21]`, `HOSPICE_IDS = [13, 14]` — discharge codes meaning the patient died or went to hospice (from `IDS_mapping.csv`).
- `ID_COLS`, `CODE_ID_COLS`, `NUMERIC_COLS` — column groups used by EDA and feature construction.
- `PALETTE` — the colors for every figure (blue, orange, aqua, yellow, magenta and greys), chosen to stay distinguishable for color-blind readers.

### `_env_int(name, default)`
Reads environment variable `name` as an integer, or returns `default` if it is not set. Used to build the configuration constants.

### `gpu_available()`
Returns `True` if a GPU should be used. If `USE_GPU` is set, it obeys it. Otherwise it checks whether the `nvidia-smi` program exists and runs successfully (which is true only on machines with an NVIDIA GPU and driver). Used only by the XGBoost pipeline.

### `set_seed(seed=SEED)`
Seeds Python's `random`, NumPy's global generator and `PYTHONHASHSEED`. Called at the start of every script. (Models also receive their own `random_state`/`random_seed`, which is what really makes them reproducible.)

### `find_data_dir()`
Finds the folder containing `diabetic_data.csv`, in this order:
1. `READMISSION_DATA_DIR` if set;
2. on Kaggle, searches everything under `/kaggle/input/` for the file (Kaggle mounts datasets at `/kaggle/input/datasets/<user>/<slug>/`, and this search does not depend on the exact path);
3. otherwise `data/raw/` in the repo.

### `load_raw()`
Reads the raw CSV and adds the binary target.
- `na_values=["?"]` — the dataset marks missing values with `?`.
- `keep_default_na=False` — **important**: by default pandas turns the text `"None"` into a missing value. In `A1Cresult` and `max_glu_serum`, `"None"` means "test not performed", which is real information, so we switch that behaviour off.
- Adds `readmit_30d = 1` when `readmitted == "<30"`, else `0`.

### `results_path(name)`
Returns `RESULTS_DIR / name`, creating the folder if needed. Every script writes through this function, so FAST mode automatically redirects all outputs.

### `save_json(obj, name)` / `load_json(name)`
Write/read a JSON file in `results/`. `save_json` uses `_json_default` so NumPy numbers and arrays can be saved.

### `save_dataset(obj)` / `load_dataset()`
Save/load the processed dataset (a Python dictionary) with `joblib` to `data/processed/dataset.joblib`. `load_dataset` gives a clear error if `02_preprocess.py` has not been run.

### `train_test(ds)`
Splits the processed dataset dictionary into `X_train, y_train, groups_train, X_test, y_test, groups_test` using the boolean mask `ds["is_test"]`. The DataFrame indices are reset so that `.iloc` positions match NumPy positions.

### `_json_default(o)`
Converts NumPy integers, floats and arrays into plain Python types for JSON. Raises an error for anything else.

### `setup_matplotlib()`
Switches matplotlib to the non-interactive `Agg` backend (works on servers without a screen), sets 300 dpi output, font size 9, light grey grid, no top/right borders and the palette's text colors. Returns `pyplot` so scripts can use `plt`.

---

## 5. `code/features.py`

Two kinds of preprocessing live here:

- **Stateless** steps (the same for every row, nothing learned from data): run once in `02_preprocess.py` via `build_features`.
- **Fitted** steps (one-hot encoding, scaling): defined as scikit-learn `ColumnTransformer`s and placed **inside** the model pipelines, so they are re-fitted inside every cross-validation fold.

### Constants

- `MISSING = "Missing"` — label used for missing categorical values.
- `DRUG_COLS` — the 23 diabetes medication columns. Each has values `No`, `Steady`, `Up` (dose increased) or `Down` (dose decreased).
- `NEAR_CONSTANT_DRUGS` — the 15 drug columns where one value covers at least 99% of encounters (copied from `results/eda.json → near_constant_columns`). They carry almost no information and are dropped as separate features.
- `KEPT_DRUGS` — the remaining 8 drugs, kept as features.
- `UNKNOWN_IDS` — for each of the three ID columns, the codes that mean "unknown / not available / NULL / not mapped" in `IDS_mapping.csv`; they are merged into one level `unknown`.
- `DROP_COLS = ["weight"]` — dropped because ~97% missing.

### `icd9_group(code)`
Maps one ICD-9 diagnosis code to one of nine clinical groups, following Strack et al. (2014):

| Group | ICD-9 codes |
|---|---|
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Diabetes | 250.xx |
| Injury | 800–999 |
| Musculoskeletal | 710–739 |
| Genitourinary | 580–629, 788 |
| Neoplasms | 140–239 |
| Other | everything else, including codes starting with V or E |

Missing codes return `"Missing"`. Codes like `"250.83"` are converted to numbers first; `int(v) == 250` catches all `250.xx`.

### `build_features(df)`
Builds the 36-column feature table from the raw table, row by row. It never looks at other rows or at the target, so it cannot leak information. Step by step:

1. **Demographics.** `race` (missing → `"Missing"`), `gender`, and `age` converted from a band like `"[70-80)"` to its midpoint `75`, so the model sees age as a number.
2. **Admission context.** The three ID columns become strings (so they are treated as categories, not numbers — code 7 is not "more" than code 3), with unknown codes merged into `"unknown"`. `payer_code` and `medical_specialty` keep missing values as `"Missing"` (missingness itself may carry information).
3. **Numeric utilisation.** The 8 count columns from `NUMERIC_COLS` as floats, plus `total_prior_visits` = outpatient + emergency + inpatient visits in the previous year.
4. **Diagnoses.** `diag_1_group`, `diag_2_group`, `diag_3_group` via `icd9_group`.
5. **Labs.** `A1Cresult` and `max_glu_serum` as categories, including `"None"` (not tested).
6. **Medications.** The 8 kept drug columns as categories, plus four counts computed from **all 23** drugs: `n_drugs_active` (not `No`), `n_drugs_up`, `n_drugs_down`, `n_drugs_changed` (= up + down). Also `change` and `diabetesMed`.

Identifiers (`encounter_id`, `patient_nbr`) and the target are **not** in the output.

### `split_columns(X)`
Returns two lists: numeric columns (any numeric dtype) and categorical columns (everything else). With the current features: 14 numeric, 22 categorical.

### `onehot_preprocessor(num_cols, cat_cols, scale=False, min_frequency=50)`
Builds a `ColumnTransformer` used by logistic regression, random forest, XGBoost and LightGBM:
- numeric columns: passed through unchanged, or standardized (`StandardScaler`) when `scale=True` (only for logistic regression, which is sensitive to feature scale);
- categorical columns: `OneHotEncoder` turns each category into a 0/1 column. Categories seen fewer than `min_frequency=50` times in the training fold are pooled into one "infrequent" column (fewer noisy columns). Categories never seen in training (`handle_unknown="infrequent_if_exist"`) go to that infrequent column instead of causing an error.
- `sparse_threshold=0.0` forces a **dense** matrix. Reason: XGBoost treats the implicit zeros of a sparse matrix as *missing values*. That would turn real zeros (e.g. "0 prior inpatient visits") into "missing". This bug was found and fixed during development (it showed up as a failed SHAP additivity check).

### `native_preprocessor(num_cols, cat_cols)`
For CatBoost, which handles categories itself. It only selects and orders the columns, keeps the original column names (`verbose_feature_names_out=False`) and returns a pandas DataFrame (`set_output(transform="pandas")`) so CatBoost can see which columns are text.

### `original_feature(name, input_cols)`
After one-hot encoding, a feature is called e.g. `cat__diag_1_group_Circulatory`. This function maps it back to the original column (`diag_1_group`): it strips the `num__`/`cat__` prefix and returns the longest original column name that the rest starts with. "Longest" matters because some names are prefixes of others. Used by `05_explain.py` to add SHAP values of all one-hot columns of one feature.

---

## 6. `code/models.py`

Everything about models: how each one is built, how it is tuned, and the stacking estimator.

### Constants

- `MODEL_NAMES = ["lr", "rf", "xgb", "lgbm", "cat"]` — the five learners.
- `BASE_LEARNERS = ["xgb", "lgbm", "cat"]` — the three that go into the stack.
- `MODEL_LABELS` — readable names for tables and figures.
- `N_EST_MIN`, `N_EST_MAX`, `N_EST_STEP` — range for the number of trees (200–1000 in steps of 100; tiny in FAST mode).
- `DEFAULT_PARAMS` — a sensible starting configuration per model. It is always evaluated as the first Optuna trial, so tuning can only improve on it.

The two `warnings.filterwarnings` lines silence two harmless messages (feature-name warnings and an XGBoost device fallback message).

### `suggest_params(name, trial)`
Defines the **search space** of each model for Optuna. `trial.suggest_*` asks Optuna for a value in a range; `log=True` means values are sampled evenly on a log scale (good for things like learning rate where 0.01 vs 0.02 matters as much as 0.1 vs 0.2).

| Model | Tuned hyperparameters (range) |
|---|---|
| lr | `C` inverse regularization (0.001–10, log) |
| rf | `n_estimators` (200–500), `max_depth` (4–20), `min_samples_leaf` (5–100, log), `max_features` (0.05–0.5) |
| xgb | `n_estimators` (200–1000), `learning_rate` (0.01–0.2, log), `max_depth` (3–8), `min_child_weight` (1–20, log), `subsample` (0.6–1), `colsample_bytree` (0.4–1), `reg_lambda`, `reg_alpha` (0.001–10, log) |
| lgbm | `n_estimators`, `learning_rate`, `num_leaves` (15–127, log), `min_child_samples` (10–200, log), `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha` |
| cat | `iterations` (200–1000), `learning_rate`, `depth` (4–8), `l2_leaf_reg` (1–10, log), `random_strength` (0–2), `bagging_temperature` (0–1) |

### `make_pipeline(name, num_cols, cat_cols, params=None, class_weight=False, pos_weight=1.0, seed=SEED)`
Builds one complete model as `Pipeline([("pre", preprocessor), ("model", estimator)])`. Because preprocessing is inside the pipeline, every time the pipeline is fitted on a training fold, the encoder is also fitted only on that fold.

- `params` — hyperparameters (from tuning); `None` means `DEFAULT_PARAMS`.
- `class_weight=True` — turns on class weighting for the imbalance ablation:
  - logistic regression: `class_weight="balanced"`;
  - random forest: `class_weight="balanced_subsample"` (balanced per tree);
  - XGBoost and LightGBM: `scale_pos_weight=pos_weight`, where `pos_weight` = (number of negatives) / (number of positives) in the training set (about 7.7);
  - CatBoost: `auto_class_weights="Balanced"`.
- Per model:
  - `lr`: one-hot + scaling; `LogisticRegression(max_iter=3000)`.
  - `rf`: one-hot; `RandomForestClassifier(n_jobs=N_JOBS)`.
  - `xgb`: one-hot; `XGBClassifier(tree_method="hist", device="cuda" or "cpu", eval_metric="logloss")`.
  - `lgbm`: one-hot; `LGBMClassifier(subsample_freq=1, verbose=-1)`. `subsample_freq=1` is needed for LightGBM to actually use `subsample`.
  - `cat`: native preprocessor; `CatBoostNative` (below).

### `class CatBoostNative`
A thin wrapper around `CatBoostClassifier`. Why it exists: scikit-learn copies models with `clone()`, which requires that a model's constructor arguments are stored unchanged. `CatBoostClassifier` modifies its `cat_features` argument internally, so `clone()` fails. The wrapper stores only plain arguments and creates the real CatBoost model inside `fit`.

- `__init__(params, class_weight, seed)` — just stores the arguments (scikit-learn rule: no logic here).
- `fit(X, y)` — detects the categorical columns (columns with text, `dtype == object`), creates `CatBoostClassifier` with those `cat_features`, the seed, all threads, no console output and no files written to disk, fits it, and stores `classes_` (required by scikit-learn).
- `predict_proba(X)`, `predict(X)` — passed straight to the fitted CatBoost model.

### `grouped_cv(seed=SEED, n_splits=CV_FOLDS)`
Returns `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`. This splitter:
- keeps all encounters of the same patient (group) in the same fold, and
- tries to keep the positive rate similar in every fold (stratified).
Every cross-validation in the project uses this function, so the grouping is never forgotten.

### `tune(name, X, y, groups, num_cols, cat_cols, n_trials, timeout, n_folds, seed=SEED)`
Hyperparameter search with Optuna.
1. Builds the grouped folds and keeps the first `n_folds` (3) to save time.
2. `objective(trial)`: gets a candidate configuration from `suggest_params`, then for each fold fits the pipeline on the training part and computes PR-AUC (`average_precision_score`) on the validation part. After each fold it reports the running mean to Optuna; if the trial is clearly worse than earlier ones (`MedianPruner`, active after 5 trials), it is stopped early ("pruned"). Returns the mean PR-AUC.
3. Creates a study that **maximizes** the objective with the TPE sampler (a Bayesian method that proposes promising values based on past trials), seeded for reproducibility.
4. `enqueue_trial(DEFAULT_PARAMS[name])` makes the default configuration the first trial.
5. Runs until `n_trials` trials or `timeout` seconds, whichever comes first, and returns the study (`study.best_params`, `study.best_value`, `study.trials_dataframe()`).

Why PR-AUC: with 11% positives, PR-AUC focuses on how well the model finds the rare readmissions.

### `logit(p, eps=1e-6)`
Converts a probability into log-odds: `log(p / (1 - p))`. Probabilities are clipped to `[1e-6, 1 - 1e-6]` first to avoid infinity. Used as the input to the meta-learner and for SHAP checks.

### `class GroupOOFStackingClassifier`
The stacking model. scikit-learn has a `StackingClassifier`, but it cannot pass patient groups to its internal cross-validation (we tested: version 1.7 rejects `groups` even with metadata routing). Without groups, the same patient could appear in a base model's training data and in the fold it predicts, so the meta-learner would learn from overly optimistic predictions. This class fixes that.

- `__init__(estimators, final_estimator=None, cv=None)` — `estimators` is a list of `(name, pipeline)` pairs; `final_estimator` is the meta-learner (default logistic regression); `cv` is the splitter (default `grouped_cv()`).
- `fit(X, y, groups)`:
  1. For each base pipeline, `cross_val_predict(..., groups=groups, cv=cv, method="predict_proba")` produces an **out-of-fold** probability for every training row: each row is predicted by a model that never saw that patient. These are stacked into the matrix `oof_` (one column per base model).
  2. Each base pipeline is then refitted on **all** training data (`estimators_`) — these are the models used at test time.
  3. The meta-learner is fitted on `logit(oof_)` and the true labels (`final_estimator_`).
  4. Stores `classes_` and `names_`.
- `base_predict_proba(X)` — matrix of the refitted base models' probabilities.
- `predict_proba(X)` — base probabilities → logit → meta-learner → final probability. In formula: `p = sigmoid(b + w1·logit(p_xgb) + w2·logit(p_lgbm) + w3·logit(p_cat))` (Equation 1 in the paper).
- `predict(X)` — class label with a 0.5 threshold (not used for evaluation; thresholds are chosen separately).

---

## 7. `code/evaluation.py`

### Constants
- `METRICS` — the metric names reported everywhere: `roc_auc, pr_auc, brier, ece, precision, recall, f1`.
- `RECALL_TARGET = 0.70` — the recall level for the screening operating point.
- `SERIES_COLORS` — the five figure colors in a fixed order.

### `best_f1_threshold(y, p)`
Finds the probability cut-off that gives the highest F1 score. `precision_recall_curve` returns precision and recall for every possible threshold; F1 = 2·P·R/(P+R) is computed for each, and the threshold with the maximum is returned. Called on **out-of-fold** training predictions, never on test data.

### `recall_threshold(y, p, target=0.70)`
Returns the **highest** threshold whose recall is still at least 70%. "Highest" means: flag as few patients as possible while still catching 70% of readmissions. If no threshold reaches the target, returns the lowest threshold.

### `ece(y, p, n_bins=10, w=None)`
Expected calibration error. Sorts predictions, splits them into 10 groups of equal size, and in each group compares the average predicted probability with the actual readmission rate. ECE is the size-weighted average of these gaps (0 = perfectly calibrated). `w` (weights) lets the bootstrap reuse it.

### `classification_metrics(y, p, threshold, w=None)`
Computes all metrics for one set of predictions:
- threshold-free: ROC-AUC, PR-AUC (average precision), Brier score (mean squared error of probabilities), ECE;
- at the given threshold: precision, recall, F1 and `flagged_fraction` (share of patients who would get an alert);
- also returns the threshold itself.
All metrics accept weights `w`, which is how the bootstrap works without copying data.

### `bootstrap_ci(y, p, groups, threshold, n_boot, seed=SEED, alpha=0.05)`
95% confidence intervals with a **patient-level (cluster) bootstrap**:
1. Lists the unique patients (`pd.factorize`).
2. For each of `n_boot` repetitions, draws patients with replacement (a patient can be drawn 0, 1, 2… times) and gives each encounter a weight equal to how often its patient was drawn. This is equivalent to building a resampled dataset but much faster.
3. Skips a draw if it has no positives or no negatives.
4. Computes all metrics per draw, and returns the 2.5% and 97.5% quantiles of each metric.
Resampling patients rather than encounters respects the fact that encounters of the same patient are not independent.

### `metrics_with_ci(y, p, groups, threshold, n_boot, seed=SEED)`
Point estimates from `classification_metrics` plus `<metric>_ci_low` / `<metric>_ci_high` columns from `bootstrap_ci`, in one dictionary (one table row).

### `fold_metrics(y, p, fold_ids, threshold)`
Splits out-of-fold predictions by fold number and computes the metrics per fold. Returns a table with one row per fold, from which mean ± std over folds are taken.

### `fold_ids_from(cv, X, y, groups)`
Returns an array saying which fold each training row belongs to (as a validation row), using the same splitter as the out-of-fold predictions.

### `mean_std(df, by, metrics=METRICS)`
Groups a results table (e.g. by model) and returns mean and standard deviation of each metric, with columns named like `roc_auc_mean`, `roc_auc_std`.

### `fit_calibrator(kind, p, y)`
Returns a function that maps raw probabilities to calibrated probabilities:
- `"none"` — returns probabilities unchanged;
- `"isotonic"` — `IsotonicRegression`: learns a non-decreasing step function from predicted to observed rates; `out_of_bounds="clip"` handles test values outside the training range;
- `"sigmoid"` — Platt scaling: a logistic regression on the logit of the probability (with almost no regularization, `C=1e6`), i.e. it learns a new slope and intercept on the log-odds scale.
The calibrator is fitted on out-of-fold training predictions and then applied to test predictions.

### `plot_roc_pr(plt, y, preds, fname, title_suffix="")`
Two-panel figure: ROC curves (left) and precision-recall curves (right) for several models, with the AUC values in the legend. The dashed line shows chance level (diagonal for ROC; the positive rate for PR).

### `plot_reliability(plt, y, preds, fname, n_bins=10)`
Reliability (calibration) diagram: for 10 equal-size groups, mean predicted probability (x) vs. observed readmission rate (y). Points on the dashed diagonal mean perfect calibration. The Brier score of each model is in the legend. The axes are zoomed to the range where predictions actually lie.

### `plot_hbar(plt, labels, values, fname, xlabel, top=20)`
Horizontal bar chart of the top 20 items (used for SHAP feature importance).

---

## 8. `code/01_eda.py`

Exploratory data analysis. Reads only the raw CSV. Everything it writes is descriptive (no model).

### Constants
- `LAB_COLS` — the two lab columns where `"None"` means not tested.
- `NEAR_CONSTANT_SHARE = 0.99` — threshold for "near-constant" columns.

### `wilson_ci(k, n, z=1.96)`
95% Wilson score confidence interval for a proportion `k/n` (e.g. readmission rate in an age group). It behaves better than the simple "p ± 1.96·SE" formula for small groups or rates near 0.

### `rate_table(df, col, max_level=None)`
For every value of column `col`, counts encounters and positives, computes the readmission rate and its Wilson interval. `max_level` caps the values (e.g. `number_inpatient` ≥ 5 grouped as "5+"). Missing values get their own row.

### `plot_rate(plt, table, col, overall, fname, xlabel, horizontal=False, labels=None)`
Bar chart of a `rate_table` with error bars for the confidence intervals and a dashed line for the overall rate. `horizontal=True` makes horizontal bars (used for discharge codes, which have many levels).

### `main()`
1. Loads the data.
2. **Target:** counts of the original 3 classes, number of positives, and which encounters end in death/hospice.
3. **Patients:** encounters per patient, and each patient's first encounter (by `encounter_id`).
4. **Missingness:** fraction of `?` per column → `eda_missingness.csv`; fraction of `"None"` in the lab columns.
5. **Cardinality:** for each categorical column, the number of distinct values and the share of the most common one → `eda_cardinality.csv`; columns with top share ≥ 99% are listed as near-constant.
6. **Numeric summary:** mean, std, quartiles of the count columns → `eda_numeric_summary.csv`.
7. **Readmission rate** by age, prior inpatient visits and discharge code → three CSVs.
8. **`eda.json`** with all headline numbers (these are what the paper quotes): sizes, encounters per patient and share from repeat patients, target counts and rate, death/hospice counts and the rate after excluding them, first-encounter rate, missingness, lab "not tested" shares, near-constant columns, number of distinct ICD-9 codes.
9. **Figures:** class balance, missingness, and the three readmission-rate charts.

---

## 9. `code/02_preprocess.py`

Builds the modelling dataset and the patient-level train/test split.

### Constants
- `EXCLUDE_EXPIRED` — from the environment (default on).
- `N_OUTER_FOLDS = 5` — the test set is one fifth of the patients.

### `main()`
1. Loads the raw data.
2. **Cohort:** removes encounters discharged as expired or to hospice (they cannot be readmitted), unless `EXCLUDE_EXPIRED=0`.
3. **FAST mode only:** keeps a random subset of whole patients (about 15% of rows). Sampling whole patients keeps the grouping meaningful.
4. Sorts by `encounter_id` so row order is deterministic.
5. Builds `X = build_features(df)`, `y`, and `groups = patient_nbr`.
6. **Split:** `StratifiedGroupKFold(5, shuffle=True, random_state=42)`; the first fold becomes the test set (`is_test`). Same patient never on both sides; similar positive rate in both.
7. **Leakage check:** `assert` that the set of test patients and the set of training patients do not overlap. The script stops with an error if they ever do.
8. Saves `dataset.joblib` with `X`, `y`, `groups`, `is_test`, `encounter_id`, and the numeric/categorical column lists.
9. Writes `split.json` (sizes, patients, positives and rates for train and test, dropped columns, feature counts, and `patients_in_both_train_and_test = 0`) and `features.csv` (each feature with its type and number of levels).

`part(mask)` is a small inner function that summarizes one side of the split.

---

## 10. `code/03_baselines.py`

Tunes and evaluates every single model, and runs the class-weight ablation.

### `main()`
1. Loads the dataset and splits it into train/test. Computes `pos_weight` = negatives / positives in training. Builds the grouped 5-fold splitter and the fold number of every training row.
2. **Tuning** (for each of the 5 models): calls `tune(...)`, saves every trial to `tuning_trials_<model>.csv`, and writes the best hyperparameters plus metadata (best CV PR-AUC, how many trials completed/pruned, seconds used) to `best_params.json`. The file is rewritten after each model, so a crash still leaves partial results.
3. **Evaluation** (for each model, without and with class weights):
   - out-of-fold probabilities on the training set with `cross_val_predict` (grouped folds);
   - threshold = best F1 on those out-of-fold probabilities;
   - per-fold metrics → mean ± std (`baselines_cv.csv`);
   - refit on the whole training set, predict the test set once;
   - unweighted models: test metrics with bootstrap confidence intervals (`baselines_test.csv`);
   - both variants side by side (CV and test) in `imbalance_ablation.csv`.
4. Saves all out-of-fold and test probabilities to `data/processed/preds_baselines.joblib` for later analysis.

---

## 11. `code/04_stacking.py`

The main experiment: single boosted models vs. simple average vs. stack, with calibration and two operating points, repeated over several seeds.

### `CALIBRATIONS = ["none", "sigmoid", "isotonic"]`
The three calibration variants compared.

### `meta_learner()`
Returns a fresh `LogisticRegression(C=1.0)` — the stacking meta-learner (mild regularization; it only has 3 inputs).

### `main()`
1. Loads the dataset and the tuned hyperparameters from `best_params.json`.
2. For each seed in `SEEDS` (42, 43, 44):
   1. New grouped folds for this seed; base pipelines (XGBoost, LightGBM, CatBoost) with this seed.
   2. Fits `GroupOOFStackingClassifier` → base out-of-fold probabilities (`oof_base`), base test probabilities (`te_base`), stack test probabilities (`te_stack`).
   3. **Stack out-of-fold probabilities** (`oof_stack`): the meta-learner itself is run through the same grouped folds on `oof_base`. Needed because the stack's own training-set predictions would be overfitted, and thresholds/calibrators must be fitted on honest predictions.
   4. Five candidates: each base model, the simple average of the three, and the stack — each as a pair (out-of-fold, test).
   5. For each candidate and each calibration: fit the calibrator on out-of-fold, apply to both; pick the best-F1 threshold on calibrated out-of-fold; compute test metrics. For uncalibrated predictions also compute metrics at the 70%-recall threshold (operating points).
   6. **Only for seed 42:** save the fitted stacker (for SHAP), the meta-learner weights (`meta_weights_seed42.csv`), the correlation between base models' out-of-fold logits (`base_oof_correlation_seed42.csv` — high correlation means little to gain from stacking), test metrics with bootstrap CIs for every candidate (`stacking_test_ci_seed42.csv`), the ROC/PR figure and the reliability figure.
   7. Prints a one-line summary per seed.
3. Writes the summary tables: `stacking_by_seed.csv` (every seed × candidate × calibration), `stacking_comparison.csv` (uncalibrated, mean ± std over seeds — the main ablation table), `calibration_comparison.csv` (Brier/ECE/AUC per calibration), and `operating_points.csv`.

---

## 12. `code/05_explain.py`

SHAP explanations for each base model and, exactly, for the stack.

### `N_SHAP`
Number of test encounters explained (2,000; 300 in FAST mode). SHAP on all 19,718 would be slow and gives the same picture.

### `shap_values(name, pipe, X)`
Returns SHAP values (in log-odds) and the transformed feature table for one base pipeline:
1. Applies the pipeline's own preprocessor to `X` (so SHAP explains exactly what the model sees), converts to a dense DataFrame with the transformed column names.
2. CatBoost: uses CatBoost's built-in exact SHAP (`get_feature_importance(type="ShapValues")`); the last column it returns is the expected value and is removed.
3. XGBoost/LightGBM: `shap.TreeExplainer(model).shap_values(...)`. Handles older output formats (a list of two arrays, or a 3-D array) by taking the positive class.

### `grouped(sv, columns, input_cols)`
Adds up the SHAP values of all one-hot columns that belong to the same original feature (using `original_feature`). Because SHAP values are additive, the sum is the contribution of the whole feature. After this, all three models are described with the same 36 features.

### `main()`
1. Loads the test set, the seed-42 stacker and a random sample of `N_SHAP` test encounters.
2. Picks the best single base model by mean test PR-AUC from `stacking_comparison.csv`.
3. For each base model with meta-weight `w`:
   - SHAP values → per-column importance (`shap_importance_<model>.csv`) and per-feature importance (`shap_importance_grouped_<model>.csv`), measured as mean absolute SHAP value;
   - adds `w × grouped SHAP` to the stack's attribution (Equation 2 in the paper: because the stack is linear in the base log-odds, these weighted sums are the stack's exact SHAP values);
   - **additivity check:** the centred log-odds of the model must equal the centred sum of its SHAP values (centring removes the constant expected value);
   - for the best model: SHAP beeswarm plot and top-20 bar chart.
4. The same additivity check for the whole stack; all maximum deviations are saved in `shap_additivity_check.json` (values around 1e-6 or smaller mean "exact").
5. Stack importance → `shap_importance_stack.csv` and `shap_top_features_stack.png`.

---

## 13. `tools/kaggle_nb.py`

Turns the scripts in `code/` into a Kaggle notebook, pushes it, waits for it and downloads the results. The code in `code/` stays the only copy; the notebook is regenerated every time.

### Constants
- `REPO_ROOT`, `CODE_DIR`, `BUILD_DIR` (`build/kaggle/`), `OUTPUT_DIR` (`build/kaggle_outputs/`), `CONFIG_FILE` (`tools/kaggle_config.json`).
- `KAGGLE_WORKDIR = "/kaggle/working"` — where Kaggle notebooks run and save outputs.
- `REPORT_PACKAGES` — libraries whose versions are recorded in `env_kaggle.json`.

### `load_config()` / `_username_from_kaggle_json()`
Read `tools/kaggle_config.json`. If its `username` is empty, take it from `KAGGLE_USERNAME` or from `~/.kaggle/kaggle.json` (only the username is read, never the key).

### `numbered_scripts()` / `helper_modules()`
List `code/*.py` files that start with a digit (pipeline steps) or not (helpers).

### `resolve_scripts(names, use_all)`
Turns user input like `01`, `01_eda` or `01_eda.py` into script paths; `--all` selects every numbered script. Exits with a clear message if a name matches zero or several files.

### `default_slug(cfg, scripts, use_all)`
Builds the notebook's URL name, e.g. `readmission-01-eda` or `readmission-full` (max 50 characters).

### `_cell(cell_type, source)`
Creates one notebook cell (markdown or code) in Jupyter's JSON format, with a stable `id` derived from its content (newer notebook formats require cell ids).

### `_git_info()`
Returns the short git commit hash, with `-dirty` added if files in `code/` have uncommitted changes. Recorded in the notebook so every Kaggle result can be traced to the exact code.

### `build_notebook(scripts, cfg, args)`
Assembles the notebook:
1. a markdown header (scripts, commit, build time);
2. optional `!pip install` cell (`--pip`);
3. a setup cell: moves to `/kaggle/working`, creates `code/` and `results/`, adds `code/` to Python's import path, applies `--env` settings, and writes `results/env_kaggle.json` (CPU count, platform, Python and library versions, GPU name, commit, settings);
4. one `%%writefile code/<file>.py` cell per helper module and selected script — this recreates the `code/` folder on Kaggle;
5. `--include` files (e.g. `results/best_params.json`) recreated the same way;
6. one `%run code/<script>.py` cell per selected script — `%run` stops the notebook with an error if a script fails;
7. a final `!ls -la results` cell.

### `build(args, cfg)`
Resolves scripts and slug (validates the slug format), writes `build/kaggle/<slug>/notebook.ipynb` and `kernel-metadata.json`. The metadata tells Kaggle the notebook id (`<username>/<slug>`), that it is private (unless `--public`), whether GPU and internet are on, and attaches the dataset from the config.

### `kaggle(*args, capture=False)`
Runs the `kaggle` command-line tool with the given arguments; exits with an error message if the command fails.

### `kernel_id_for(args, cfg)`
Works out the notebook id from `--slug` or from the script names (used by `status` and `fetch`).

### `status(kernel_id)`
Runs `kaggle kernels status` and turns its output into a simple word: `running`, `complete`, `error`, ….

### `fetch(kernel_id, slug, sync)`
Downloads everything the notebook wrote (`kaggle kernels output`) into `build/kaggle_outputs/<slug>/`. With `sync`, copies its `results/` folder into the repo's `results/`.

### `wait(kernel_id, poll, max_failures=10)`
Checks the status every `poll` seconds until the run is complete, failed or cancelled. Network errors are retried up to 10 times in a row instead of stopping (added after a Wi-Fi drop interrupted a wait).

### `main()`
Command-line interface with four commands:
- `build` — only create the notebook folder;
- `push` — build and upload (which starts the run); `--wait` then waits and downloads with sync;
- `status` — print the run status;
- `fetch` — download outputs (`--sync` copies results into the repo).
Shared options: script names or `--all`, `--slug`. Build/push options: `--gpu`, `--internet`, `--pip`, `--public`, `--include`, `--env`.

---

## 14. Other files

- **`tools/kaggle_config.json`** — Kaggle username (empty = read from `kaggle.json`), the dataset id attached to every notebook (`vedeshp/diabetes-130-us-hospitals-uci-296`), the notebook name prefix (`readmission`) and the notebook title.
- **`kaggle_dataset/dataset-metadata.json`** — description of the private Kaggle dataset (title, id, license CC BY 4.0, source and citation). Used once with `kaggle datasets create -p kaggle_dataset`. The CSV copies in that folder are git-ignored.
- **`requirements.txt`** — libraries for running locally (Kaggle already has them).
- **`.gitignore`** — keeps raw/processed data (`data/`), the Kaggle CSV copies, build folders (`build/`) and FAST-mode results (`results_fast/`) out of git.
- **`CLAUDE.md`**, **`PROJECT_BRIEF.md`**, **`README.md`** — project rules, the project brief, and how to run things.
- **`paper/`** — the draft, references, IEEE guide and literature review.

---

## 15. Output files and who writes them

| File in `results/` | Written by | Content |
|---|---|---|
| `eda.json`, `eda_*.csv`, `eda_*.png` | 01 | Dataset description |
| `split.json`, `features.csv` | 02 | Split sizes and leakage check; feature list |
| `best_params.json`, `tuning_trials_<model>.csv` | 03 | Tuned hyperparameters; every Optuna trial |
| `baselines_cv.csv` | 03 | Out-of-fold metrics, mean ± std over 5 folds |
| `baselines_test.csv` | 03 | Test metrics with 95% CIs (no class weights) |
| `imbalance_ablation.csv` | 03 | With vs. without class weights |
| `stacking_by_seed.csv` | 04 | Every seed × model × calibration |
| `stacking_comparison.csv` | 04 | Main ablation table (mean ± std over seeds) |
| `calibration_comparison.csv` | 04 | Brier / ECE / AUC by calibration method |
| `operating_points.csv` | 04 | Metrics at 70% recall |
| `stacking_test_ci_seed42.csv` | 04 | Test metrics with 95% CIs |
| `meta_weights_seed42.csv` | 04 | Meta-learner weights |
| `base_oof_correlation_seed42.csv` | 04 | Correlation of base models' predictions |
| `roc_pr_curves_seed42.png`, `reliability_seed42.png` | 04 | Figures |
| `shap_importance_*.csv`, `shap_*.png` | 05 | Feature importance and plots |
| `shap_additivity_check.json` | 05 | Proof that stack SHAP values are exact |
| `env_kaggle.json` | Kaggle notebook setup cell | Software/hardware of the run |

Large intermediate files in `data/processed/` (not in git): `dataset.joblib` (02), `preds_baselines.joblib` (03), `stacker_seed42.joblib` (04).

---

## 16. Known limitations and things to watch

- **Tuning is time-limited.** With a 900 s budget, the number of completed trials depends on machine speed (CatBoost completed only 4 in the first run). Results are reproducible on the same machine type, not necessarily trial-for-trial elsewhere. Raise `TUNE_TIMEOUT` for CatBoost if you want a fairer comparison.
- **The CV estimate in `baselines_cv.csv` is slightly optimistic**, because tuning used three of the same five folds. The held-out test set is not affected; report test numbers as the main result.
- **Thresholded metrics per fold** (`baselines_cv.csv`) use one threshold chosen on all out-of-fold predictions, so precision/recall/F1 per fold are slightly optimistic. The test-set numbers use the threshold honestly.
- **Seed repetitions share one test set**, so "± std over seeds" measures model/fold randomness, not test-sample uncertainty — that is what the bootstrap CIs are for.
- **CatBoost on GPU is not used.** XGBoost on GPU can give slightly different numbers than on CPU, so keep the device fixed between runs you compare.
- **`encounter_id` order** is used only to sort rows and to define "first encounter" in EDA; it is assumed (not documented) to follow time.
- **Pickled models depend on library versions** (e.g. the scikit-learn 1.6.1 stacker cannot be loaded with 1.7.1). Re-run 04 rather than moving `.joblib` files between environments.

## 17. How to change common things

- **Different budgets:** `python tools/kaggle_nb.py push --all --env N_TRIALS=80 TUNE_TIMEOUT=1800 N_SEEDS=5 --wait`.
- **Quick local test:** set `FAST=1` and run the scripts; outputs go to `results_fast/`.
- **Sensitivity analysis with death/hospice encounters:** run with `EXCLUDE_EXPIRED=0` (write to a separate copy of `results/` so the main results are not overwritten).
- **Add a model:** add it to `MODEL_NAMES`, `MODEL_LABELS`, `DEFAULT_PARAMS`, `suggest_params` and `make_pipeline` in `models.py`; to stack it, also add it to `BASE_LEARNERS`.
- **Add a feature:** add it in `build_features` (row-wise only, no target); it is picked up automatically as numeric or categorical by `split_columns`.
- **Re-run only stacking and SHAP on Kaggle with the existing tuned parameters:** `python tools/kaggle_nb.py push 02 04 05 --include results/best_params.json --wait`.
