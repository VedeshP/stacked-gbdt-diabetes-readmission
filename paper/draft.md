Patient-Level Evaluation of Gradient-Boosted Ensembles for Diabetic Readmission Prediction

Vedesh Pandya
vedeshskp@gmail.com

Abstract—Unplanned 30-day readmission is a key quality indicator, and diabetic inpatients are a large, high-risk group. We predict readmission on the public Diabetes 130-US Hospitals dataset (101,766 encounters from 71,518 patients; 11.16% positive). Because 46.2% of encounters belong to patients admitted more than once, we split the data by patient and keep this grouping in every internal step, including the out-of-fold predictions that train the stacking meta-learner. We add features derived from each patient's earlier encounters and compare tuned logistic regression, random forest, XGBoost, LightGBM and CatBoost with simple averaging and a logistic-regression stack of the three boosted models, and evaluate class weighting, calibration and out-of-fold thresholds. On a held-out test set of 14,038 unseen patients, the stack reaches an area under the receiver operating characteristic curve of 0.695 (95% confidence interval 0.680-0.708) and an area under the precision-recall curve of 0.259 (prevalence 0.111), averaged over three runs. Patient-history features provide most of the gain, raising the first measure for a tuned LightGBM model from 0.682 to 0.693, whereas stacking adds only about 0.002 over the best single model, CatBoost, because the base models' predictions are highly correlated (0.92 to 0.95). The boosted models are already well calibrated (Brier score 0.092), and class weighting worsens calibration without improving discrimination. Exact Shapley-value attributions for the stack identify discharge disposition, prior inpatient visits, payer code and the outcome of the previous encounter as the strongest predictors. This retrospective study uses de-identified public data and is not intended for clinical use.

Index Terms—Clinical risk prediction, data leakage, diabetes mellitus, gradient boosting, hospital readmission, probability calibration, SHAP, stacked generalization.

I. INTRODUCTION

Readmission shortly after discharge is costly and often signals gaps in care at the transition from hospital to home. Among Medicare fee-for-service beneficiaries, 19.6% were rehospitalized within 30 days, and unplanned rehospitalizations cost Medicare an estimated 17.4 billion US dollars in 2004 [1]. Since 2012, the Hospital Readmissions Reduction Program has reduced payments to hospitals with excess 30-day readmissions [2], which has increased interest in tools that flag high-risk patients at discharge so that follow-up resources can be targeted.

Patients with diabetes are an important group: diabetes is common among inpatients, often complicates the main reason for admission, and is associated with a higher readmission risk [3]. Strack et al. [4] analyzed a large multi-hospital database of diabetic encounters, and the de-identified data were released as the Diabetes 130-US Hospitals dataset [5], now a common benchmark. Prediction is difficult: most readmission models perform poorly [6], and these data contain no vital signs, clinical notes or post-discharge information. Gradient-boosted trees such as XGBoost [7], LightGBM [8] and CatBoost [9] are strong learners for such tabular data, and stacked generalization [10] can combine them. Recent studies continue to apply such models, explained with SHapley Additive exPlanations (SHAP), to diabetic readmission [11], [12], [13].

Three methodological issues limit how far published results on this dataset can be trusted and compared. First, the data record encounters, not patients: 46.2% of encounters come from patients admitted more than once. A random split of encounters places the same patient in both training and test sets, a form of leakage known to inflate reported performance [14], [15]. In stacked ensembles the issue recurs, because the out-of-fold predictions that train the meta-learner must also be patient-grouped, and the stacking estimator of scikit-learn [16] (version 1.7) does not accept group labels. Second, calibration is rarely reported, although a risk score used to allocate resources must give probabilities that match observed rates [17], and reporting guidelines ask for it explicitly [18]. Third, explanations of a single ensemble member do not describe what the ensemble as a whole relies on.

A further question is what information the models receive. Prior use of hospital services is among the most common predictors in readmission models [6], yet studies on this dataset treat every encounter as independent and ignore the patient's own earlier stays, which are known at a later discharge.

We revisit the task with a protocol that addresses these issues and report the results as they are, including where gains are small. Our contributions are:

1) A leakage-free protocol that extends patient-grouped evaluation, previously used for single models [19], to every internal step of a stacked ensemble: tuning, out-of-fold stacking, calibration and threshold selection.

2) Patient-history features computed only from each patient's earlier encounters, which raise the test ROC-AUC of a tuned LightGBM model from 0.682 to 0.693 under the same protocol.

3) A controlled comparison of five tuned learners, simple averaging and stacking, with ablations for features, class weighting and calibration, reported over repeated runs with patient-level bootstrap confidence intervals.

4) Exact SHAP attributions [20], [21] for the stacked model, obtained by combining the base models' SHAP values with the meta-learner's weights, and an open, reproducible pipeline.

II. RELATED WORK

Kansagara et al. [6] found that most readmission risk models discriminate poorly and that few use measures of function, illness severity or social factors. On the Diabetes 130-US Hospitals data, Strack et al. [4] used logistic regression for inference rather than prediction: they grouped International Classification of Diseases, Ninth Revision (ICD-9) codes into clinical categories and showed that whether glycated hemoglobin (HbA1c) was measured was associated with readmission. Later studies report results that are hard to compare because outcome definitions, splits and metrics differ. Bhuvan et al. [22] compared five classifiers on a random 75%/25% split of encounters and reported an area under the precision-recall curve (PR-AUC) of 0.242 for random forest. Mingle [23] reported values of the area under the receiver operating characteristic curve (ROC-AUC) from 0.65 to 0.79 for age-specific ensembles on a random split, without an overall figure. Shang et al. [24] down-sampled the majority class and reported a ROC-AUC of 0.661 for random forest on an 80%/20% split, and Emi-Johnson and Nkrumah [25] reported 0.667 for XGBoost on a stratified 80%/20% split. None of these studies describes keeping a patient's encounters in one partition. Liu et al. [19] grouped encounters by patient in 5-fold cross-validation, removed duplicate patient records and applied the synthetic minority oversampling technique (SMOTE) within the training folds, and reported ROC-AUC values of 0.64 for XGBoost and 0.63 for random forest. Zarghani [26] reported only accuracy (92.22% for LightGBM), which is uninformative at 11% prevalence, and listed the encounter and patient identifiers among the input features.

Ensembles have also been studied. Lu and Uddin [27] built an explainable stacking model with random under-sampling for this task. Salim and Ibrahim [11] combined logistic regression, random forest, XGBoost and LightGBM in a stacking ensemble with Platt scaling and reported a nested cross-validated ROC-AUC of 0.664, a calibrated ROC-AUC of 0.688 and a Brier score of 0.094; they note that their cross-validation did not enforce patient-level splits. Afridi et al. [13] proposed a leakage-resistant framework in which resampling is restricted to the training folds. In other clinical domains, patient-level and temporal leakage has been shown to inflate published results [15]. To our knowledge, no study on this dataset evaluates a stacked ensemble with patient-grouped out-of-fold predictions while also using the patient's earlier encounters as features.

The three boosted learners differ in design: XGBoost [7] grows regularized trees level by level, LightGBM [8] grows trees leaf-wise on histogram bins, and CatBoost [9] builds symmetric trees and encodes categories with ordered target statistics. Such differences can produce partly different errors, the condition under which stacking [10] helps. Their hyperparameters are commonly tuned with Optuna [28] and the tree-structured Parzen estimator [29]. Platt scaling [30] and isotonic regression [31] are standard post-hoc calibration methods, and boosted models can produce distorted probabilities that benefit from them [32]. The uncertainty of performance estimates can be quantified by bootstrap resampling [33], which must respect clustering when patients contribute several encounters. SHAP [20] with exact tree algorithms [21] is the most common way to explain boosted models.

III. DATASET AND PREPROCESSING

A. Data and Outcome

The Diabetes 130-US Hospitals dataset (CC BY 4.0) contains 101,766 inpatient encounters of 71,518 diabetic patients from 130 US hospitals (1999-2008), with 50 attributes covering demographics, admission and discharge, utilization counts, three ICD-9 codes, laboratory results, 23 diabetes medications and prior-year visits. The positive class is readmission within 30 days; readmission after 30 days and no readmission are negative. Overall, 11,357 encounters (11.16%) are positive. We exclude the 2,423 encounters ending in death or hospice (discharge codes 11, 13, 14, 19, 20, 21), of which only 43 were positive, leaving 99,343 encounters with a positive rate of 11.39%. Table I summarizes the cohort and the split.

[Insert Table I here: results/table1_cohort.csv]

TABLE I
Cohort and Patient-Level Split

B. Encounter Features

Weight (96.9% missing) is dropped; medical specialty (49.1% missing) and payer code (39.6% missing) keep missingness as a category. For HbA1c and glucose, the value "None" means the test was not performed (83.3% and 94.7% of encounters) and is kept as a category, because ordering the test is itself informative. Each ICD-9 diagnosis code is mapped to one of nine clinical groups (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms and other) and is also kept as a three-character code for finer detail. Fifteen near-constant medication columns are removed, and counts of active medications and dosage changes are added, together with age-band midpoints and a total of prior-year visits.

C. Patient-History Features

The dataset does not contain dates, so we use the encounter identifier as the time order within each patient. This order is consistent with the labels: of the 30,248 encounters that have an earlier encounter of the same patient, only 1.6% follow an encounter that was labeled as not readmitted (see results/encounter_order_check.json). For each encounter we compute, from the same patient's earlier encounters only, the number of earlier encounters, the number of earlier 30-day readmissions, the outcome label and length of stay of the previous encounter, and its discharge disposition. These quantities are known at the current discharge; no later information is used, and the history is computed before any exclusion. The final feature set has 44 features, 17 numeric and 27 categorical (see results/features.csv). All steps in this section are row-wise; fitted transformations are learned inside the model pipelines (Section IV).

D. Patient-Level Split

Patients have up to 40 encounters; 23.5% have more than one, accounting for 46.2% of encounters. We split by patient with stratified group 5-fold partitioning (seed 42) and hold out one fold as the test set: 79,625 training encounters from 55,952 patients (positive rate 11.46%) and 19,718 test encounters from 14,038 patients (11.10%), with no patient in both. All later steps use patient-grouped folds of the training set, and the test set is used once.

IV. METHODOLOGY

The complete pipeline is shown in Fig. 1. Everything inside the dashed region is fitted on training patients only.

[Insert Fig. 1 here, double-column width: paper/figures/fig1_methodology.png]

Fig. 1. Proposed pipeline. Steps inside the dashed region are fitted on training patients only; the test set is used once for evaluation.

A. Base Learners

We use L2-regularized logistic regression, random forest, XGBoost, LightGBM and CatBoost, each wrapped with its preprocessing in one scikit-learn pipeline so that every encoder is fitted within the training folds only. For the first four, categorical features are one-hot encoded with rare levels (fewer than 50 occurrences) pooled, and numeric features are standardized for logistic regression only. The encoded matrix is kept dense, because XGBoost treats the implicit zeros of a sparse matrix as missing values. CatBoost uses native categorical handling, which suits the high-cardinality diagnosis codes. Logistic regression and random forest serve as reference models.

B. Patient-Grouped Out-of-Fold Stacking

The training set is split into K = 5 stratified, patient-grouped folds. Each base learner m is fitted on K - 1 folds and predicts the held-out fold, giving an out-of-fold probability p_im for every training encounter from a model that never saw that patient. A logistic-regression meta-learner combines the log-odds:

p_i,stack = sigma(b + w_1 logit(p_i1) + w_2 logit(p_i2) + w_3 logit(p_i3)),   (1)

where sigma is the logistic function and the intercept b and weights w_m are fitted on the out-of-fold predictions. The log-odds scale keeps the meta-learner small (four parameters), makes it a joint Platt-type recalibration and allows exact explanations (Section IV-D). The base learners are then refitted on the full training set, and (1) is applied to their test predictions. We implement this as a separate estimator that receives the patient identifiers, and evaluate the simple average of the three base probabilities as an ablation.

Thresholds and calibrators are fitted on out-of-fold probabilities; for the stack, these are obtained by passing the meta-learner itself through the same patient-grouped folds.

C. Imbalance, Calibration and Thresholds

About 11% of training encounters are positive. We compare no reweighting with class weighting, in which positives receive a weight equal to the negative-to-positive ratio of the training set (scale_pos_weight for XGBoost and LightGBM, balanced class weights for the other learners). We avoid synthetic oversampling because it distorts predicted probabilities. Each model is evaluated uncalibrated, with Platt scaling, which fits a logistic regression to the log-odds, and with isotonic regression, which fits a non-decreasing step function; both are fitted on out-of-fold probabilities and applied unchanged to the test set. An alert requires a threshold, which we choose on out-of-fold probabilities at two operating points: the maximum F1 score, and the highest threshold that still reaches a recall of 0.70, a screening setting in which most readmissions should be flagged.

D. Explaining the Stack

We compute exact TreeSHAP values for each base learner; they are additive in the log-odds. Summing the values of all one-hot columns of a feature maps every model to the same 44 features. With phi_ijm the SHAP value of feature j for encounter i under learner m, (1) implies

logit(p_i,stack) = b + sum_m w_m E[logit(p_im)] + sum_j sum_m w_m phi_ijm,   (2)

so sum_m w_m phi_ijm is an exact attribution for the stack rather than a surrogate approximation. On 2,000 test encounters, the largest deviation between the stack's log-odds and the sum of these attributions was 4.4e-06 (see results/shap_additivity_check.json). We report mean absolute attributions over the same encounters.

V. EXPERIMENTAL SETUP

Metrics. We report ROC-AUC, PR-AUC (average precision; a model without skill scores the prevalence, 0.111), the Brier score and the expected calibration error (ECE, 10 equal-frequency bins), and precision, recall, F1 and the flagged fraction at the chosen thresholds. Accuracy is not reported.

Tuning. Hyperparameters are tuned on the training set with Optuna using the tree-structured Parzen estimator (seed 42) and a median pruner, maximizing mean PR-AUC over three of the five grouped folds, with a default configuration as the first trial and a budget of 40 trials or 900 s per learner. The search spaces cover tree count, learning rate, depth or leaves, child size, subsampling and L1/L2 penalties for the boosted models, and the main capacity parameters of the reference models. Because the budget is time-limited, the number of completed trials depends on the machine; completed trials were 28 (logistic regression), 4 (random forest), 11 (XGBoost), 30 (LightGBM) and 4 (CatBoost). The selected values are stored with the code (results/best_params.json).

Protocol. Single learners are evaluated by 5-fold grouped out-of-fold predictions (mean ± std over folds) and once on the test set. The ensemble comparison is repeated with seeds 42, 43 and 44 and reported as mean ± std; since the test set is fixed, this spread reflects model and fold randomness. Sampling uncertainty is given by 95% confidence intervals (CIs) from a patient-level cluster bootstrap with 1,000 resamples, in which patients rather than encounters are drawn. The feature ablation re-uses the tuned LightGBM configuration on nested feature sets.

Implementation. Python 3.13 with scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, Optuna 5.0.0 and SHAP 0.52.0, run as one Kaggle notebook on 4 CPU cores without a GPU; the complete pipeline took 251 minutes (see results/env_kaggle.json and results/runtime_kaggle.json). The code, the notebook builder and all result files are available at https://github.com/VedeshP/stacked-gbdt-diabetes-readmission (commit c4a8ddf).

VI. RESULTS AND DISCUSSION

A. Model Comparison

Table II reports test performance for all models, and Fig. 2 shows the corresponding ROC and precision-recall curves. Logistic regression reaches a ROC-AUC of 0.677 and random forest 0.684. The three boosted models are close to each other: XGBoost 0.690, LightGBM 0.693 and CatBoost 0.693, with CatBoost giving the highest PR-AUC among single models (0.259). The stack reaches 0.695 ± 0.000 and a PR-AUC of 0.259, and the simple average reaches 0.695 and 0.258. Cross-validated values on the training set (0.687 to 0.689 for the boosted models) agree with the test values.

[Insert Table II here: results/table2_main_results.csv]

TABLE II
Test-Set Performance (Mean ± Std Over Three Runs for Boosted Models and Ensembles; 95% Patient-Level Bootstrap CI for ROC-AUC)

[Insert Fig. 2 here, double-column width: paper/figures/fig2_roc_pr_curves.png]

Fig. 2. ROC (left) and precision-recall (right) curves on the test set for the boosted models and ensembles (seed 42). Legend values are ROC-AUC and PR-AUC; dashed lines show chance level.

The gain from ensembling is therefore small. The stack improves on CatBoost by about 0.002 in ROC-AUC and does not differ from the simple average, and the 95% confidence intervals of all boosted models and both ensembles overlap almost completely (stack 0.680-0.708, CatBoost 0.678-0.706). The reason is visible in the out-of-fold predictions: the log-odds of the three base learners are correlated between 0.92 and 0.95, so they make largely the same errors and there is little for the meta-learner to combine. The meta-learner weights (0.39 for XGBoost, 0.07 for LightGBM and 0.60 for CatBoost) mainly select CatBoost and XGBoost, consistent with LightGBM adding little once the other two are present. The spread of the ROC-AUC over the three repeated runs is below 0.001 for every model, which is much smaller than the test-set sampling uncertainty given by the bootstrap intervals.

Our test ROC-AUC values lie at the upper end of the range reported for this dataset in Section II, although most of those studies split encounters rather than patients and are not directly comparable. The only patient-grouped study we found reported 0.63 to 0.64, below all of our models, including logistic regression.

B. Effect of Features

The feature ablation in Table III explains most of the improvement over a standard pipeline. With encounter features only, the tuned LightGBM model reaches a ROC-AUC of 0.682 (95% CI 0.667-0.696) and a PR-AUC of 0.244. Adding three-character diagnosis codes raises the ROC-AUC to 0.686, adding the patient-history features raises it to 0.691, and using both gives 0.693 and a PR-AUC of 0.254. The history features thus contribute more than finer diagnosis coding, and more than the choice between boosted models or the use of stacking. This agrees with the observation in Section II that prior use of hospital services is among the most common predictors in readmission models, and it shows that, on this dataset, what the model is allowed to know matters more than how complex it is.

[Insert Table III here: results/table3_ablations.csv]

TABLE III
Ablations on the Test Set: Feature Sets (LightGBM), Class Weighting and Calibration of the Stack

C. Class Imbalance and Calibration

Class weighting does not help (Table III). It changes the test ROC-AUC of the five learners by between -0.004 and 0.000, but raises the Brier score from 0.092-0.094 to 0.163-0.219, because the weighted models systematically overestimate risk. Without weighting, all boosted models are already well calibrated. Fig. 3 shows that the reliability curves of CatBoost and of the stack lie close to the diagonal. For the stack, Platt scaling leaves the Brier score (0.0919) and ECE (0.0050) essentially unchanged, which is expected because the meta-learner already applies a logistic transformation to the log-odds. Isotonic regression lowers the ECE slightly to 0.0044 but reduces the PR-AUC from 0.259 to 0.251, because its step function merges many distinct probabilities into ties. We therefore keep the uncalibrated stack.

[Insert Fig. 3 here, single-column width: paper/figures/fig3_reliability.png]

Fig. 3. Reliability diagram on the test set (ten equal-frequency bins). The uncalibrated stack and the Platt-scaled stack coincide.

D. Operating Points

At the threshold that maximizes F1 on the out-of-fold predictions, the stack reaches a precision of 0.218, a recall of 0.468 and an F1 score of 0.298 on the test set. At the screening threshold chosen for a recall of 0.70, the test recall is 0.693 and the precision 0.170, with 45.2% of encounters flagged. In practical terms, catching seven in ten readmissions requires following up almost half of all discharged patients, and about one in six flagged patients is readmitted, compared with one in nine without a model. This trade-off, rather than the ROC-AUC alone, decides whether such a model is useful.

E. Explaining the Stack

Fig. 4 shows the mean absolute SHAP attribution of each feature for the stacked model. The strongest predictors are discharge disposition, the number of inpatient visits in the previous year, payer code, the outcome of the previous encounter and the three-character primary diagnosis, followed by the total number of prior visits and the number of earlier encounters. Four of the seven leading features describe the patient's past use of the hospital, which is consistent with the feature ablation. Payer code is likely a proxy for insurance type and socioeconomic status; its influence should be examined for fairness before any use of such a model. Because the stack is linear in the base log-odds, these attributions are exact: the largest deviation between the stack's log-odds and the sum of the attributions was 4.4e-06 over 2,000 test encounters.

[Insert Fig. 4 here, single-column width: paper/figures/fig4_shap_stack.png]

Fig. 4. Mean absolute SHAP attribution of the 15 most important features for the stacked model on 2,000 test encounters. Asterisks mark patient-history features.

VII. LIMITATIONS AND CONCLUSION

This study has several limitations. The data were collected between 1999 and 2008 in US hospitals, and clinical practice, coding and readmission policy have changed since then. All results come from one data source with a single held-out test set and no external validation. The dataset has no dates, so the patient-history features rely on the encounter identifier as a time order; the label check supports this assumption but cannot prove it, and stays at hospitals outside the dataset remain invisible. Encounters ending in death or hospice were excluded, and tuning was limited by time, so random forest and CatBoost completed only four tuning trials each. Finally, the model is intended as a research benchmark, not for clinical use.

Under a patient-level protocol, 30-day readmission in this dataset remains only moderately predictable, with a best test ROC-AUC of 0.695. Using information from each patient's earlier encounters improved discrimination more than any modeling choice, while stacking three highly correlated boosted models added almost nothing over the best single model or a simple average. The boosted models were well calibrated without post-hoc adjustment, and class weighting harmed calibration. The linear stack does, however, allow exact explanations at little cost. Future work should validate these findings on more recent data from other hospitals, add information that is missing here, such as laboratory values, vital signs and social factors, and study the fairness of predictors such as payer code.

ACKNOWLEDGMENT

The author used Claude (Anthropic, model Claude Opus 5.5) to help write and debug the analysis code, to generate Fig. 1, to search for and check literature, and to draft the text of all sections. The author reviewed, verified and edited all AI-assisted content and takes full responsibility for the paper.

REFERENCES

See paper/references.md.
