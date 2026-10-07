Patient-Grouped, Calibrated Stacking of Gradient-Boosted Trees for 30-Day Readmission Prediction in Diabetic Inpatients

Vedesh Pandya

Abstract—Unplanned readmission within 30 days of discharge is a widely used indicator of hospital care quality and a target of financial penalties, and diabetic inpatients are a large, high-risk group. We study 30-day readmission prediction on the public Diabetes 130-US Hospitals dataset (101,766 encounters from 71,518 patients, 1999-2008), where 11.16% of encounters are followed by readmission within 30 days. Because 46.2% of encounters belong to patients with more than one admission, we split the data by patient, so that no patient contributes to both training and testing, and we extend this grouping to every internal cross-validation step, including the out-of-fold predictions used to train the stacking meta-learner. We compare tuned logistic regression, random forest, XGBoost, LightGBM and CatBoost with a simple average and a logistic-regression stack of the three boosted models, and we evaluate class weighting, Platt and isotonic calibration, and decision thresholds chosen only on out-of-fold predictions. On a held-out test set of 19,718 encounters from 14,038 unseen patients, the stacked model reaches a ROC-AUC of TODO and a PR-AUC of TODO (prevalence 0.111), compared with TODO for the best single model, with a Brier score of TODO after calibration. Because the meta-learner is linear in the base models' log-odds, we obtain exact SHAP attributions for the full stack; the strongest predictors are TODO (from results/shap_importance_stack.csv). Discrimination remains modest, which is consistent with the limited clinical detail in the data. This is a retrospective study on de-identified public data and the model is not intended for clinical use.

Index Terms—hospital readmission, diabetes mellitus, gradient boosting, stacked generalization, probability calibration, data leakage, SHAP, clinical risk prediction.

I. INTRODUCTION

Readmission to hospital shortly after discharge is costly for health systems and often signals gaps in care at the transition from hospital to home. In a study of Medicare fee-for-service claims, Jencks et al. found that 19.6% of discharged beneficiaries were rehospitalized within 30 days, and estimated the cost to Medicare of unplanned rehospitalizations in 2004 at 17.4 billion US dollars [1]. Since 2012, the Hospital Readmissions Reduction Program of the Centers for Medicare and Medicaid Services has reduced payments to hospitals with higher than expected 30-day readmission rates for selected conditions [2]. These incentives have increased interest in tools that identify, at the time of discharge, the patients most likely to return, so that limited follow-up resources such as early clinic visits, medication reconciliation and post-discharge telephone calls can be directed to them.

Patients with diabetes are an important group for this problem. Diabetes is common among hospitalized patients, it is frequently a secondary diagnosis that complicates the main reason for admission, and hospitalized patients with diabetes may face a higher risk of readmission than those without it [3]. Strack et al. [4] analyzed a large multi-hospital database of diabetic inpatient encounters and reported that HbA1c was measured in only 18.4% of encounters and that the association between HbA1c measurement and readmission depended on the primary diagnosis. The de-identified data from that study were released as the Diabetes 130-US Hospitals dataset [5] and have since become a common benchmark for readmission prediction.

Predicting readmission is difficult. A systematic review of readmission risk models concluded that most of them performed poorly [6], and the Diabetes 130-US Hospitals data contain no vital signs, clinical notes or post-discharge information. Gradient-boosted decision trees such as XGBoost [7], LightGBM [8] and CatBoost [9] are strong learners for tabular data of this kind, and stacked generalization [10] can combine several of them. However, three methodological issues limit how much published results on this dataset can be trusted and compared.

The first issue is leakage between training and test data. The dataset records encounters, not patients, and in our analysis 46.2% of all encounters come from patients who were admitted more than once. A random or stratified split of encounters places different admissions of the same patient in both the training and the test set, so that the model can partly recognize patients rather than learn transferable risk patterns. Leakage of this kind is a well-documented source of over-optimistic results in machine-learning-based science [11]. For stacked ensembles the problem appears a second time, because the out-of-fold predictions used to train the meta-learner must also be generated with patient-grouped folds; the stacking estimator of scikit-learn [19] (version 1.7) does not accept group labels for its internal cross-validation. The second issue is that many studies report discrimination (accuracy or ROC-AUC) but not calibration, although a risk score used to allocate resources must give probabilities that match observed rates [12]. The third issue is that ensembles of boosted trees are difficult to interpret, which limits clinical trust.

In this paper we revisit 30-day readmission prediction for diabetic inpatients with an evaluation protocol designed to avoid these problems, and we report the results as they are, including where the gains are small. Our contributions are as follows.

1) A leakage-free evaluation protocol in which the test set contains only unseen patients and every internal step (hyperparameter search, out-of-fold stacking, calibration and threshold selection) uses patient-grouped, stratified folds on the training data only.

2) A controlled comparison of tuned logistic regression, random forest, XGBoost, LightGBM and CatBoost against simple averaging and a logistic-regression stack of the three boosted models, with ablations for class weighting and for the ensembling strategy, reported as mean and standard deviation over repeated runs and with patient-level bootstrap confidence intervals.

3) An assessment of probability quality using reliability curves, the Brier score and the expected calibration error, before and after Platt and isotonic calibration, together with decision thresholds chosen on out-of-fold predictions for a maximum-F1 and a fixed-recall operating point.

4) Exact SHAP attributions [13], [14] for the stacked model, obtained by combining the base models' tree SHAP values with the meta-learner's coefficients, which explains the ensemble rather than only one of its members.

5) A fully reproducible pipeline with a fixed random seed, released as open-source code (TODO: add repository URL and commit hash).

The rest of the paper is organized as follows. Section II reviews related work. Section III describes the dataset and preprocessing. Section IV presents the methodology, Section V the experimental setup, and Section VI the results. Section VII discusses limitations and concludes.

II. RELATED WORK

A. Readmission Risk Prediction

Kansagara et al. [6] reviewed 26 readmission risk models and found that most had poor discrimination. Most models included comorbidity and prior use of medical services, while few included measures of overall health, function, illness severity or social determinants of health, and the authors called for further work to improve performance [6]. For patients with diabetes, Rubin [3] summarized risk factors for readmission that include lower socioeconomic status, racial or ethnic minority status, comorbidity burden, public insurance, emergent or urgent admission and a recent prior hospitalization.

B. Studies on the Diabetes 130-US Hospitals Dataset

Strack et al. [4] introduced the data and used multivariable logistic regression to study the relationship between HbA1c measurement and early readmission, rather than to build a predictive model. Later work has treated the dataset as a machine-learning benchmark. Emi-Johnson and Nkrumah [15] compared logistic regression, random forest, XGBoost and a deep neural network and reported the highest ROC-AUC for XGBoost (0.667), followed by logistic regression (0.642) and random forest (0.630). Their data were divided into training and test sets of 80% and 20% using stratified sampling; the description does not state that encounters of the same patient were kept together, so these figures are not directly comparable with patient-level results. TODO: add two or three further verified studies on this dataset, noting for each how the split was made and whether calibration was reported.

C. Boosting, Stacking and Calibration in Clinical Prediction

Gradient-boosted trees are widely used for tabular clinical data. XGBoost [7] grows trees level-wise with a regularized objective, LightGBM [8] grows trees leaf-wise with histogram-based splits, and CatBoost [9] uses symmetric trees and ordered target statistics for categorical features. These design differences can lead to partly different errors, which is the condition under which stacked generalization [10] can improve on its members. Calibration determines whether predicted risks can be read as probabilities; Van Calster et al. [12] argue that poor calibration can make an otherwise discriminative model misleading for clinical decisions, and Platt scaling [16] and isotonic regression [17] are standard post-hoc remedies whose behavior on boosted models was compared by Niculescu-Mizil and Caruana [18]. For interpretation, SHAP values [13] and their exact, efficient computation for tree ensembles [14] have become the most common way to explain boosted models in clinical studies.

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

REFERENCES

See paper/references.md.
