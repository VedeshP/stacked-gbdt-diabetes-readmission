Patient-Grouped, Calibrated Stacking of Gradient-Boosted Trees for 30-Day Readmission Prediction in Diabetic Inpatients

Vedesh Pandya

Abstract—Unplanned readmission within 30 days of discharge is a widely used indicator of hospital care quality, and diabetic inpatients are a large, high-risk group. We study 30-day readmission prediction on the public Diabetes 130-US Hospitals dataset (101,766 encounters from 71,518 patients, 1999-2008; 11.16% positive). Because 46.2% of encounters belong to patients admitted more than once, we split the data by patient and extend this grouping to every internal step, including the out-of-fold predictions that train the stacking meta-learner. We compare tuned logistic regression, random forest, XGBoost, LightGBM and CatBoost with simple averaging and a logistic-regression stack of the three boosted models, and evaluate class weighting, Platt and isotonic calibration, and thresholds chosen only on out-of-fold predictions. Over three repeated runs on a held-out test set of 19,718 encounters from 14,038 unseen patients, the stack reaches a ROC-AUC of 0.683 and a PR-AUC of 0.244 (prevalence 0.111), which is practically indistinguishable from simple averaging (0.683, 0.243) and from the best single model, LightGBM (0.684, 0.243); the base models' out-of-fold predictions are highly correlated (0.95 to 0.98). The boosted models are already well calibrated (Brier score 0.093), post-hoc calibration changes little, and class weighting worsens the Brier score without improving discrimination. Exact SHAP attributions for the stack identify prior inpatient visits, discharge disposition and total prior visits as the strongest predictors. Discrimination remains modest, consistent with the limited clinical detail in the data. This retrospective study uses de-identified public data and the model is not intended for clinical use.

Index Terms—Clinical risk prediction, data leakage, diabetes mellitus, gradient boosting, hospital readmission, probability calibration, SHAP, stacked generalization.

I. INTRODUCTION

Readmission to hospital shortly after discharge is costly for health systems and often signals gaps in care at the transition from hospital to home. In a study of Medicare fee-for-service claims, Jencks et al. found that 19.6% of discharged beneficiaries were rehospitalized within 30 days, and estimated the cost to Medicare of unplanned rehospitalizations in 2004 at 17.4 billion US dollars [1]. Since 2012, the Hospital Readmissions Reduction Program of the Centers for Medicare and Medicaid Services has reduced payments to hospitals with higher than expected 30-day readmission rates for selected conditions [2]. These incentives have increased interest in tools that identify, at the time of discharge, the patients most likely to return, so that limited follow-up resources such as early clinic visits, medication reconciliation and post-discharge telephone calls can be directed to them.

Patients with diabetes are an important group for this problem. Diabetes is common among hospitalized patients, it is frequently a secondary diagnosis that complicates the main reason for admission, and hospitalized patients with diabetes may face a higher risk of readmission than those without it [3]. Strack et al. [4] analyzed a large multi-hospital database of diabetic inpatient encounters and reported that HbA1c was measured in only 18.4% of encounters and that the association between HbA1c measurement and readmission depended on the primary diagnosis. The de-identified data from that study were released as the Diabetes 130-US Hospitals dataset [5] and have since become a common benchmark for readmission prediction.

Predicting readmission is difficult. A systematic review of readmission risk models concluded that most of them performed poorly [6], and the Diabetes 130-US Hospitals data contain no vital signs, clinical notes or post-discharge information. Gradient-boosted decision trees such as XGBoost [7], LightGBM [8] and CatBoost [9] are strong learners for tabular data of this kind, and stacked generalization [10] can combine several of them. However, three methodological issues limit how much published results on this dataset can be trusted and compared.

The first issue is leakage between training and test data. The dataset records encounters, not patients, and in our analysis 46.2% of all encounters come from patients who were admitted more than once. A random or stratified split of encounters places different admissions of the same patient in both the training and the test set, so that the model can partly recognize patients rather than learn transferable risk patterns. Leakage of this kind is a well-documented source of over-optimistic results in machine-learning-based science [11]. For stacked ensembles the problem appears a second time, because the out-of-fold predictions used to train the meta-learner must also be generated with patient-grouped folds; the stacking estimator of scikit-learn [12] (version 1.7) does not accept group labels for its internal cross-validation. The second issue is that many studies report discrimination (accuracy or ROC-AUC) but not calibration, although a risk score used to allocate resources must give probabilities that match observed rates [13]. The third issue is that ensembles of boosted trees are difficult to interpret, which limits clinical trust.

In this paper we revisit 30-day readmission prediction for diabetic inpatients with an evaluation protocol designed to avoid these problems, and we report the results as they are, including where the gains are small. Our contributions are as follows.

1) A leakage-free evaluation protocol that extends patient-grouped evaluation, used for single models in [14], to every internal step of a stacked ensemble: the test set contains only unseen patients, and hyperparameter search, out-of-fold stacking, calibration and threshold selection all use patient-grouped, stratified folds on the training data only.

2) A controlled comparison of tuned logistic regression, random forest, XGBoost, LightGBM and CatBoost against simple averaging and a logistic-regression stack of the three boosted models, with ablations for class weighting and for the ensembling strategy, reported as mean and standard deviation over repeated runs and with patient-level bootstrap confidence intervals.

3) An assessment of probability quality using reliability curves, the Brier score and the expected calibration error, before and after Platt and isotonic calibration, together with decision thresholds chosen on out-of-fold predictions for a maximum-F1 and a fixed-recall operating point.

4) Exact SHAP attributions [15], [16] for the stacked model, obtained by combining the base models' tree SHAP values with the meta-learner's coefficients, which explains the ensemble rather than only one of its members.

5) A fully reproducible pipeline with a fixed random seed, released as open-source code (TODO: add repository URL and commit hash).

The rest of the paper is organized as follows. Section II reviews related work. Section III describes the dataset and preprocessing. Section IV presents the methodology, Section V the experimental setup, and Section VI the results. Section VII discusses limitations and concludes.

II. RELATED WORK

A. Readmission Risk Prediction

Kansagara et al. [6] reviewed 26 readmission risk models and found that most had poor discrimination. Most models included comorbidity and prior use of medical services, while few included measures of overall health, function, illness severity or social determinants of health, and the authors called for further work to improve performance [6]. For patients with diabetes, Rubin [3] summarized risk factors for readmission that include lower socioeconomic status, racial or ethnic minority status, comorbidity burden, public insurance, emergent or urgent admission and a recent prior hospitalization.

B. Studies on the Diabetes 130-US Hospitals Dataset

Strack et al. [4] introduced the data and used multivariable logistic regression to study the relationship between HbA1c measurement and early readmission, rather than to build a predictive model. Later work has treated the dataset as a machine-learning benchmark, with results that are hard to compare because of differences in outcome definition, data splitting and metrics. Bhuvan et al. [17] compared five classifiers on a random 75%/25% split of encounters and reported an area under the precision-recall curve of 0.242 for random forest when separating readmission within 30 days from all other outcomes. Mingle [18] built age-specific ensembles on data split at random into 75% training and 25% test data and reported ROC-AUC values of 0.79, 0.70 and 0.65 for the age groups 0-29, 30-69 and 70-99 years. Shang et al. [19] used random forest, naive Bayes and a tree ensemble with down-sampling of the majority class and an 80%/20% split, and reported a best ROC-AUC of 0.661 for random forest. Emi-Johnson and Nkrumah [20] reported the highest ROC-AUC for XGBoost (0.667), followed by logistic regression (0.642) and random forest (0.630), on a stratified 80%/20% split. None of these four studies describes keeping the encounters of one patient in the same partition, so their figures are not directly comparable with patient-level results. Liu et al. [14] did group encounters by patient in 5-fold cross-validation, removed duplicate patient records and applied SMOTE within the training folds, and reported ROC-AUC values of 0.64 for XGBoost and 0.63 for random forest. Some reports on this dataset give only accuracy-based figures; for example, Zarghani [21] reported 92.22% accuracy for LightGBM on a 70%/30% split, while the feature importance table of the same report lists the encounter and patient identifiers among the input features, and no ROC-AUC was given. Among the studies above, we found no report of calibration and no stacked ensemble evaluated with patient-grouped out-of-fold predictions.

C. Boosting, Stacking and Calibration in Clinical Prediction

Gradient-boosted trees are widely used for tabular clinical data. XGBoost [7] grows trees level-wise with a regularized objective, LightGBM [8] grows trees leaf-wise with histogram-based splits, and CatBoost [9] uses symmetric trees and ordered target statistics for categorical features. These design differences can lead to partly different errors, which is the condition under which stacked generalization [10] can improve on its members. Calibration determines whether predicted risks can be read as probabilities; Van Calster et al. [13] argue that poor calibration can make an otherwise discriminative model misleading for clinical decisions, and Platt scaling [22] and isotonic regression [23] are standard post-hoc remedies whose behavior on boosted models was compared by Niculescu-Mizil and Caruana [24]. For interpretation, SHAP values [15] and their exact, efficient computation for tree ensembles [16] have become the most common way to explain boosted models in clinical studies.

III. DATASET AND PREPROCESSING

A. Data Source

We use the Diabetes 130-US Hospitals for Years 1999-2008 dataset [5], released under a CC BY 4.0 license, which was extracted for the study of Strack et al. [4]. It contains 101,766 inpatient encounters of 71,518 unique patients with diabetes, described by 50 attributes covering demographics, admission and discharge details, length of stay, counts of laboratory tests, procedures, medications and diagnoses, three ICD-9 diagnosis codes, HbA1c and serum glucose test results, 23 diabetes medications and the number of outpatient, emergency and inpatient visits in the year before the encounter. The data are de-identified, and this is a retrospective study on public data.

B. Outcome Definition

The original label records readmission within 30 days, after more than 30 days, or no readmission. We define the positive class as readmission within 30 days and treat the other two values as negative. In the full dataset, 11,357 of 101,766 encounters (11.16%) are positive (Table I, see results/eda.json).

C. Cohort Selection

Encounters that ended in death or discharge to hospice (discharge disposition codes 11, 13, 14, 19, 20 and 21) cannot meaningfully be followed by a readmission, and only 43 of these 2,423 encounters were labeled positive. We exclude them, which leaves 99,343 encounters with a positive rate of 11.39%. A sensitivity analysis that keeps these encounters is TODO.

D. Missing Values and Feature Construction

Missing values are marked with a question mark in the source file. Weight is missing for 96.9% of encounters and is dropped; medical specialty (49.1% missing) and payer code (39.6% missing) are kept, with missingness as a separate category (Fig. 1, see results/eda_missingness.png). For HbA1c and maximum serum glucose, the value "None" means that the test was not performed; this applies to 83.3% and 94.7% of encounters respectively, and we keep it as a category rather than treating it as missing, because whether the test was ordered is itself informative [4]. The three diagnosis codes are mapped to nine clinical groups (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms and other) following Strack et al. [4]. Fifteen medication columns in which a single value covers at least 99% of encounters are removed, and from all 23 medication columns we derive the number of active medications and the number of dosage increases, decreases and changes. Age bands are converted to their midpoints, unknown admission and discharge codes are merged, and a total count of prior visits is added. The final feature set has 36 features, 14 numeric and 22 categorical (see results/features.csv). Encoding of categorical features is fitted inside each cross-validation fold as part of the model pipeline, so no statistics from validation or test data enter the training of any model.

E. Patient-Level Split

Patients in the dataset have between 1 and 40 encounters; 23.5% of patients have more than one encounter, and these patients account for 46.2% of all encounters (see results/eda.json). We therefore split by patient identifier using stratified group 5-fold partitioning with a fixed seed of 42 and hold out one fold as the test set. The training set contains 79,625 encounters from 55,952 patients (positive rate 11.46%) and the test set 19,718 encounters from 14,038 patients (positive rate 11.10%); no patient appears in both (Table II, see results/split.json). All model selection, stacking, calibration and threshold selection use patient-grouped folds within the training set only, and the test set is used once for the final evaluation.

IV. METHODOLOGY

A. Problem Formulation

Each encounter i is described by a feature vector x_i and a label y_i, where y_i = 1 if the patient is readmitted within 30 days of discharge and y_i = 0 otherwise. Each encounter also carries a patient identifier g_i that is used only to form the data partitions. A model outputs an estimated probability p_i = P(y_i = 1 | x_i), and an alert is raised when p_i exceeds a decision threshold t. Because the alert is meant to direct limited follow-up resources, we evaluate both how well p_i ranks patients and whether p_i agrees with observed readmission rates.

B. Base Learners and Pipelines

We use five learners: L2-regularized logistic regression, random forest, XGBoost [7], LightGBM [8] and CatBoost [9]. Each learner is wrapped together with its preprocessing in a single scikit-learn pipeline [12], so that every fitted transformation is learned from the training part of a fold only. For logistic regression, random forest, XGBoost and LightGBM, categorical features are one-hot encoded, levels seen fewer than 50 times are pooled into one infrequent level, and numeric features are passed through unchanged (standardized for logistic regression). The encoded matrix is stored in dense form, because XGBoost treats the implicit zeros of a sparse matrix as missing values, which would turn genuine zero counts, such as zero prior inpatient visits, into missing entries. CatBoost receives the categorical features as strings and handles them with its built-in ordered target statistics [9]. XGBoost, LightGBM and CatBoost are the base learners of the ensemble; logistic regression and random forest serve as reference models.

C. Patient-Grouped Out-of-Fold Stacking

Stacked generalization [10] trains a meta-learner on predictions that the base learners make for data they were not trained on. The training set is partitioned into K = 5 folds with stratified group k-fold partitioning on the patient identifier, so that all encounters of a patient fall in the same fold and each fold has a similar positive rate. For each base learner m and each fold k, the learner is fitted on the other K - 1 folds and predicts the encounters of fold k. Collecting these predictions gives an out-of-fold probability p_im for every training encounter, and no prediction is made by a model that saw any encounter of the same patient. The meta-learner is a logistic regression on the log-odds of the base predictions:

p_i,stack = sigma(b + w_1 logit(p_i1) + w_2 logit(p_i2) + w_3 logit(p_i3)),   (1)

where sigma is the logistic function and the intercept b and weights w_m are fitted on the out-of-fold predictions. After the meta-learner is fitted, each base learner is refitted on the full training set, and (1) is applied to the predictions of the refitted models at test time. Because the stacking estimator of scikit-learn does not pass group labels to its internal cross-validation, we implement this procedure as a separate estimator that receives the patient identifiers explicitly. As an ablation we also evaluate the simple average of the three base probabilities, which needs no meta-learner.

Thresholds and calibrators must be fitted on predictions that were not themselves fitted to the same labels. For single models and for the average, the out-of-fold probabilities satisfy this directly. For the stack, we obtain cross-fitted out-of-fold probabilities by running the meta-learner through the same patient-grouped folds on the matrix of base out-of-fold predictions.

D. Class Imbalance

About 11% of training encounters are positive. We compare training without reweighting against class weighting, in which positive examples receive a weight equal to the ratio of negative to positive training examples (the scale_pos_weight parameter of XGBoost and LightGBM, and balanced class weights for logistic regression, random forest and CatBoost). We do not use synthetic oversampling, because it changes the class distribution seen by the model and therefore distorts the predicted probabilities, which we want to remain interpretable as risks.

E. Probability Calibration

We compare three variants for every model: no calibration, Platt scaling [22], which fits a logistic regression to the log-odds of the predicted probability, and isotonic regression [23], which fits a non-decreasing step function. Both calibrators are fitted on the out-of-fold probabilities of the training set and then applied unchanged to the test predictions.

F. Decision Thresholds

A probability becomes an alert only after a threshold is chosen. We select thresholds on the out-of-fold probabilities of the training set at two operating points: the threshold that maximizes the F1 score, and the highest threshold that still reaches a recall of at least 0.70, a screening-oriented setting in which most readmissions should be flagged. The selected thresholds are then applied once to the test set.

G. Explaining the Stack

For each base learner we compute exact SHAP values [15] with TreeSHAP [16]. These values are additive in the model's raw output, which for XGBoost, LightGBM and CatBoost is the log-odds of the predicted probability. For the one-hot encoded models, the SHAP values of all indicator columns that come from the same original feature are summed, which preserves additivity and maps all three models to the same 36 features. Let phi_ijm be the SHAP value of feature j for encounter i under base learner m. Because (1) is linear in the base log-odds, the log-odds of the stack decompose exactly as

logit(p_i,stack) = b + sum_m w_m E[logit(p_im)] + sum_j sum_m w_m phi_ijm,   (2)

so sum_m w_m phi_ijm is an exact attribution of feature j for the stacked model rather than an approximation by a surrogate model. We confirmed this additivity numerically on 500 test encounters (see results/shap_additivity_check.json). Global importance is reported as the mean absolute attribution over a random sample of 2,000 test encounters.

V. EXPERIMENTAL SETUP

A. Evaluation Metrics

We report the area under the receiver operating characteristic curve (ROC-AUC), the area under the precision-recall curve computed as average precision (PR-AUC), the Brier score, and the expected calibration error (ECE) with 10 equal-frequency bins. At the selected thresholds we report precision, recall, F1 score and the fraction of encounters flagged. Because the positive class is rare, PR-AUC is more informative than accuracy; a classifier without skill has a PR-AUC equal to the test prevalence of 0.111. Accuracy is not reported.

B. Hyperparameter Search

The hyperparameters of each learner are tuned on the training set only with Optuna [25], using its tree-structured Parzen estimator sampler [26] with a fixed seed of 42 and a median pruner. The objective is the mean PR-AUC over the first three of the five patient-grouped folds. The default configuration of each learner is evaluated as the first trial, and the search stops after 40 trials or 900 seconds per learner, whichever comes first. The search spaces cover the number of trees, learning rate, tree depth or number of leaves, minimum child size, row and column subsampling and L1 and L2 regularization for the boosted models; tree depth, leaf size, number of trees and feature fraction for random forest; and the inverse regularization strength for logistic regression. Within the time limit, the number of completed trials was 27 for logistic regression, 9 for random forest, 21 for XGBoost, 29 for LightGBM and 4 for CatBoost, and the remaining started trials were pruned. The selected hyperparameters are listed in Table III (see results/best_params.json; all trials in results/tuning_trials_*.csv).

C. Evaluation Protocol

Each tuned learner is evaluated in two ways. First, 5-fold patient-grouped out-of-fold predictions on the training set give a cross-validated estimate, reported as mean and standard deviation over the five folds. Second, the learner is refitted on the full training set and evaluated once on the held-out test set. For the ensemble comparison, the complete procedure (fold assignment, base learner training, out-of-fold stacking, calibration and threshold selection) is repeated with three seeds (42, 43 and 44), and test metrics are reported as mean and standard deviation over the repetitions. The test set is identical in all repetitions, so this spread reflects model and fold randomness rather than sampling of the test population. To quantify sampling uncertainty, we compute 95% confidence intervals with a patient-level cluster bootstrap [27] of the test set with 1,000 resamples, in which patients rather than encounters are drawn with replacement.

D. Implementation and Hardware

All experiments are implemented in Python 3.13 with scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, Optuna 5.0.0 and SHAP 0.52.0 (see results/env_kaggle.json). The full pipeline runs as a single Kaggle notebook on CPU without a GPU accelerator; the number of CPU cores and the total run time were not recorded in this run (TODO). The random seed is fixed to 42 for data partitioning, tuning and model training. Because the tuning budget is limited by time, the number of completed trials can differ between machines, so the trials actually run are stored with the results. The code and all result files are available in the project repository (TODO: add URL and commit hash). Reporting follows the TRIPOD+AI guideline for clinical prediction models where applicable [28].

REFERENCES

See paper/references.md.
