Patient-Grouped, Calibrated Stacking of Gradient-Boosted Trees for 30-Day Readmission Prediction in Diabetic Inpatients

Vedesh Pandya

Abstract—Unplanned 30-day readmission is a key indicator of hospital care quality, and diabetic inpatients are a large, high-risk group. We study readmission prediction on the public Diabetes 130-US Hospitals dataset (101,766 encounters from 71,518 patients; 11.16% positive). Because 46.2% of encounters belong to patients admitted more than once, we split the data by patient and keep this grouping in every internal step, including the out-of-fold predictions that train the stacking meta-learner. We compare tuned logistic regression, random forest, XGBoost, LightGBM and CatBoost with simple averaging and a logistic-regression stack of the three boosted models, and evaluate class weighting, Platt and isotonic calibration, and thresholds chosen on out-of-fold predictions. On a held-out test set of 14,038 unseen patients, averaged over three runs, the stack reaches a ROC-AUC of 0.683 and a PR-AUC of 0.244 (prevalence 0.111), practically identical to simple averaging (0.683, 0.243) and to LightGBM alone (0.684, 0.243), because the base models' predictions are highly correlated (0.95 to 0.98). The boosted models are already well calibrated (Brier score 0.093); post-hoc calibration changes little, and class weighting worsens calibration without improving discrimination. Exact SHAP attributions for the stack identify prior inpatient visits, discharge disposition and total prior visits as the strongest predictors. This retrospective study uses de-identified public data and is not intended for clinical use.

Index Terms—Clinical risk prediction, data leakage, diabetes mellitus, gradient boosting, hospital readmission, probability calibration, SHAP, stacked generalization.

I. INTRODUCTION

Readmission shortly after discharge is costly and often signals gaps in care at the transition from hospital to home. Among Medicare fee-for-service beneficiaries, 19.6% were rehospitalized within 30 days, and unplanned rehospitalizations cost Medicare an estimated 17.4 billion US dollars in 2004 [1]. Since 2012, the Hospital Readmissions Reduction Program has reduced payments to hospitals with excess 30-day readmissions [2], which has increased interest in tools that flag high-risk patients at discharge so that follow-up resources can be targeted.

Patients with diabetes are an important group: diabetes is common among inpatients, often complicates the main reason for admission, and is associated with a higher readmission risk [3]. Strack et al. [4] analyzed a large multi-hospital database of diabetic encounters, and the de-identified data were released as the Diabetes 130-US Hospitals dataset [5], now a common benchmark. Prediction is difficult: most readmission models perform poorly [6], and these data contain no vital signs, clinical notes or post-discharge information. Gradient-boosted trees such as XGBoost [7], LightGBM [8] and CatBoost [9] are strong learners for such tabular data, and stacked generalization [10] can combine them.

Three methodological issues limit how far published results on this dataset can be trusted and compared. First, the data record encounters, not patients: 46.2% of encounters come from patients admitted more than once. A random split of encounters places the same patient in both training and test sets, a form of leakage known to inflate results [11]. In stacked ensembles the issue recurs, because the out-of-fold predictions that train the meta-learner must also be patient-grouped, and the stacking estimator of scikit-learn [12] (version 1.7) does not accept group labels. Second, calibration is rarely reported, although a risk score used to allocate resources must give probabilities that match observed rates [13]. Third, ensembles of boosted trees are hard to interpret.

We revisit the task with a protocol that addresses these issues and report the results as they are, including where gains are small. Our contributions are:

1) A leakage-free protocol that extends patient-grouped evaluation, previously used for single models [14], to every internal step of a stacked ensemble: tuning, out-of-fold stacking, calibration and threshold selection.

2) A controlled comparison of five tuned learners, simple averaging and stacking, with ablations for class weighting and calibration, reported over repeated runs with patient-level bootstrap confidence intervals.

3) Exact SHAP attributions [15], [16] for the stacked model, obtained by combining the base models' SHAP values with the meta-learner's weights.

4) An open, reproducible pipeline with a fixed random seed (TODO: repository URL and commit hash).

II. RELATED WORK

Kansagara et al. [6] found that most readmission risk models discriminate poorly and that few use measures of function, illness severity or social factors. On the Diabetes 130-US Hospitals data, Strack et al. [4] used logistic regression for inference on HbA1c measurement rather than prediction. Later studies report results that are hard to compare because outcome definitions, splits and metrics differ. On random encounter splits, Bhuvan et al. [17] reported a PR-AUC of 0.242 for random forest, Mingle [18] reported ROC-AUC values from 0.65 to 0.79 for age-specific ensembles, Shang et al. [19] reported a ROC-AUC of 0.661 with down-sampling, and Emi-Johnson and Nkrumah [20] reported 0.667 for XGBoost on a stratified 80%/20% split. None of these describes keeping a patient's encounters in one partition. Liu et al. [14] did group encounters by patient and reported ROC-AUC values of 0.63 to 0.64. Some reports give only accuracy; Zarghani [21] reported 92.22% accuracy for LightGBM while listing the encounter and patient identifiers among the input features. We found no study on this dataset that reports calibration or evaluates a stacked ensemble with patient-grouped out-of-fold predictions.

Calibration is often neglected in clinical prediction [13]. Platt scaling [22] and isotonic regression [23] are standard remedies, and boosted trees are known to need them in some settings [24]. SHAP [15] with exact tree algorithms [16] is the most common way to explain boosted models.

III. DATASET AND PREPROCESSING

A. Data and Outcome

The dataset [5] (CC BY 4.0) contains 101,766 inpatient encounters of 71,518 diabetic patients from 130 US hospitals (1999-2008), with 50 attributes: demographics, admission and discharge details, length of stay, counts of tests, procedures, medications and diagnoses, three ICD-9 codes, HbA1c and glucose results, 23 diabetes medications, and prior-year outpatient, emergency and inpatient visits. The positive class is readmission within 30 days; readmission after 30 days and no readmission are negative. Overall, 11,357 encounters (11.16%) are positive (Table I, see results/eda.json). We exclude the 2,423 encounters ending in death or hospice (discharge codes 11, 13, 14, 19, 20, 21), of which only 43 were positive, leaving 99,343 encounters (positive rate 11.39%). A sensitivity analysis including them is TODO.

B. Features

Weight (96.9% missing) is dropped; medical specialty (49.1% missing) and payer code (39.6% missing) keep missingness as a category (Fig. 1, see results/eda_missingness.png). For HbA1c and glucose, the value "None" means the test was not performed (83.3% and 94.7% of encounters) and is kept as a category, because ordering the test is itself informative [4]. Diagnosis codes are mapped to nine clinical groups following [4]. Fifteen near-constant medication columns are removed, and counts of active medications and dosage changes are added, together with total prior visits. The final set has 36 features, 14 numeric and 22 categorical (see results/features.csv).

C. Patient-Level Split

Patients have up to 40 encounters; 23.5% have more than one, accounting for 46.2% of encounters. We split by patient with stratified group 5-fold partitioning (seed 42) and hold out one fold as the test set: 79,625 training encounters from 55,952 patients (positive rate 11.46%) and 19,718 test encounters from 14,038 patients (11.10%), with no patient in both (Table II, see results/split.json). All later steps use patient-grouped folds of the training set, and the test set is used once.

IV. METHODOLOGY

A. Base Learners

We use L2-regularized logistic regression, random forest, XGBoost, LightGBM and CatBoost, each wrapped with its preprocessing in one scikit-learn pipeline [12] so that every encoder is fitted within the training folds only. For the first four, categorical features are one-hot encoded with rare levels (fewer than 50 occurrences) pooled, and numeric features are standardized for logistic regression only. The encoded matrix is kept dense because XGBoost treats the implicit zeros of a sparse matrix as missing, which would turn real zero counts into missing values. CatBoost uses its native categorical handling [9]. The three boosted models form the ensemble; logistic regression and random forest are reference models.

B. Patient-Grouped Out-of-Fold Stacking

The training set is split into K = 5 stratified, patient-grouped folds. Each base learner m is fitted on K - 1 folds and predicts the held-out fold, giving an out-of-fold probability p_im for every training encounter from a model that never saw that patient. A logistic-regression meta-learner combines the log-odds:

p_i,stack = sigma(b + w_1 logit(p_i1) + w_2 logit(p_i2) + w_3 logit(p_i3)),   (1)

where sigma is the logistic function and b and w_m are fitted on the out-of-fold predictions [10]. The base learners are then refitted on the full training set, and (1) is applied at test time. Because the scikit-learn stacking estimator cannot use group labels, we implement this as a separate estimator. As an ablation, we evaluate the simple average of the three base probabilities. For threshold and calibration fitting, the stack's own out-of-fold probabilities are obtained by passing the meta-learner through the same grouped folds.

C. Imbalance, Calibration and Thresholds

We compare no reweighting with class weighting, where positives receive a weight equal to the negative-to-positive ratio. We avoid synthetic oversampling because it distorts predicted probabilities. Each model is evaluated uncalibrated, with Platt scaling [22] and with isotonic regression [23], both fitted on out-of-fold probabilities. Thresholds are chosen on out-of-fold probabilities at two operating points: maximum F1, and the highest threshold reaching a recall of 0.70.

D. Explaining the Stack

We compute exact TreeSHAP values [15], [16] for each base learner; they are additive in the log-odds. Summing the values of all one-hot columns of a feature maps every model to the same 36 features. With phi_ijm the SHAP value of feature j for encounter i under learner m, (1) implies

logit(p_i,stack) = b + sum_m w_m E[logit(p_im)] + sum_j sum_m w_m phi_ijm,   (2)

so sum_m w_m phi_ijm is an exact attribution for the stack rather than a surrogate approximation. We confirmed this numerically on 500 test encounters (see results/shap_additivity_check.json) and report mean absolute attributions over 2,000 test encounters.

V. EXPERIMENTAL SETUP

Metrics. We report ROC-AUC, PR-AUC (average precision; a no-skill model scores the prevalence, 0.111), the Brier score and the expected calibration error (ECE, 10 equal-frequency bins), and precision, recall, F1 and the flagged fraction at the chosen thresholds. Accuracy is not reported.

Tuning. Hyperparameters are tuned on the training set with Optuna [25] using the TPE sampler [26] (seed 42) and a median pruner, maximizing mean PR-AUC over three of the five grouped folds, with the default configuration as the first trial and a budget of 40 trials or 900 s per learner. Completed trials were 27 (logistic regression), 9 (random forest), 21 (XGBoost), 29 (LightGBM) and 4 (CatBoost); the selected values are in Table III (see results/best_params.json).

Protocol. Single learners are evaluated by 5-fold grouped out-of-fold predictions (mean ± std over folds) and once on the test set. The ensemble comparison is repeated with seeds 42, 43 and 44 and reported as mean ± std; since the test set is fixed, this spread reflects model and fold randomness. Sampling uncertainty is given by 95% confidence intervals from a patient-level cluster bootstrap [27] with 1,000 resamples.

Implementation. Python 3.13 with scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, Optuna 5.0.0 and SHAP 0.52.0, run as one Kaggle CPU notebook without GPU (see results/env_kaggle.json; core count and run time TODO). Reporting follows TRIPOD+AI where applicable [28].

REFERENCES

See paper/references.md.
