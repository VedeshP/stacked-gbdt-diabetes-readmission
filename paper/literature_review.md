# Literature Review: 30-Day Readmission Prediction on the Diabetes 130-US Hospitals Dataset

Working document for the paper. It lists the sources reviewed, what each study did, the scores it reports, and how comparable those scores are to ours. Reference numbers in square brackets match `paper/references.md`.

Last updated: 2026-10-07.

## 1. How the review was done

- Searched with general web search, PubMed Central / Europe PMC, arXiv and publisher pages, using the dataset name, "30-day readmission", "diabetes", "machine learning", "XGBoost", "stacking" and "patient-level split".
- Included: studies that predict readmission from the UCI Diabetes 130-US Hospitals data (the Cerner Health Facts diabetic encounter extract of Strack et al.).
- For every included study, I opened the full text (HTML, Europe PMC XML or PDF) and checked four things: outcome definition, how train/test data were split, whether encounters of the same patient were kept together, and which metric was reported. When a detail is not stated in the paper, the table says "not stated". It does not mean the authors made a mistake.
- Scores are copied as reported. They are **not directly comparable** across rows because outcome definitions, splits, sampling and test sets differ.

## 2. Studies on this dataset

| # | Study | Venue | Outcome | Split / validation | Same patient kept together? | Imbalance handling | Models | Best reported score | Calibration reported? |
|---|---|---|---|---|---|---|---|---|---|
| [4] | Strack et al., 2014 | BioMed Res. Int. | <30 vs rest | Not a prediction study (inference) | n/a | n/a | Multivariable logistic regression | No predictive metric; HbA1c measured in 18.4% of encounters | n/a |
| [17] | Bhuvan et al., 2016 | arXiv | <30 vs rest (and any readmission) | Random 75%/25% split | Not stated | Not stated | NB, Bayes net, RF, AdaBoost, NN | PR-AUC 0.242 (RF) for <30 vs rest | No |
| [18] | Mingle, 2017 | Curr. Trends Biomed. Eng. Biosci. | <30 vs rest | Random 75%/25% split; 10-fold stratified CV for tuning | Not stated | Not stated | Age-specific blended ensembles | ROC-AUC 0.79 / 0.70 / 0.65 for ages 0-29 / 30-69 / 70-99; LACE baseline 0.56 | No |
| [19] | Shang et al., 2021 | BMC Med. Inform. Decis. Mak. | <30 vs rest | 80%/20% split | Not stated | Down-sampling (KNIME Equal Size Sampling) | RF, naive Bayes, tree ensemble | ROC-AUC 0.661 (RF) | No |
| [14] | Liu, Sue & Wu, 2024 | J. Med. Artif. Intell. | <30 vs rest | 5-fold group k-fold on patient | **Yes**; duplicate patient records also removed | SMOTE inside training folds | 11 models incl. RF, XGBoost, LSTM, LR | ROC-AUC 0.64 (XGBoost), 0.63 (RF); accuracy 0.88 | No |
| [21] | Zarghani, 2024 | arXiv | "readmitted" (definition not explicit) | Random 70%/30% split | Not stated | Not stated | XGBoost, LightGBM, CatBoost, DT, RF, LSTM | Accuracy 92.22%, F1 0.91 (LightGBM); no ROC-AUC | No |
| [20] | Emi-Johnson & Nkrumah, 2025 | Cureus | <30 vs rest | Stratified 80%/20% split | Not stated | None (SMOTE considered, not applied) | LR, RF, XGBoost, DNN | ROC-AUC 0.667 (XGBoost), 0.642 (LR), 0.630 (RF) | No |
| — | **This work** | — | <30 vs rest | Patient-grouped stratified 5-fold; held-out patient fold as test; grouped inner CV for tuning, stacking, calibration, thresholds | **Yes**, at every step | Class weights vs none (ablation) | LR, RF, XGBoost, LightGBM, CatBoost, average, LR stack | ROC-AUC 0.684 ± 0.001, PR-AUC 0.243 ± 0.001 (LightGBM); stack 0.683 / 0.244 | **Yes** (Brier, ECE, reliability, Platt, isotonic) |

Notes on individual studies:

- **[17] Bhuvan et al.** The paper says it "randomly split our dataset into two distinct sets", 75% and 25%. It reports PR-AUC rather than ROC-AUC for the <30-day task. Their random forest PR-AUC (0.242) is very close to ours (0.243–0.244), even though our split is patient-level and theirs is not.
- **[18] Mingle.** The paper uses the same random 75/25 split wording and reports ROC-AUC per age group. The 0.79 value is for the youngest group (0-29 years), which is a small subgroup. No overall ROC-AUC is given.
- **[19] Shang et al.** They worked in KNIME. Down-sampling balanced the classes, and the authors report that down-sampling the training set gave better results than over-sampling. A best ROC-AUC of 0.661 is reported for 30-day readmission.
- **[14] Liu, Sue & Wu.** This is the closest methodological match to our work: patient-grouped folds, removal of duplicate patient records, and SMOTE applied inside training folds only. Their ROC-AUC values (0.63–0.64) are slightly below ours. Because they also removed duplicate patients, their cohort differs from ours. Our paper must cite this study when describing patient-level evaluation, and must not claim to be the first to group by patient.
- **[21] Zarghani.** It reports only accuracy, precision, recall and F1. The SHAP feature table in the report lists `encounter_id` and `patient_nbr` among the input features. Identifier columns used as features are a known leakage risk [11]. In the paper we only state what the table shows.
- **[20] Emi-Johnson & Nkrumah.** The study used a stratified 80/20 split and did not apply SMOTE. Its best ROC-AUC is 0.667 (XGBoost).

## 3. What the comparison shows

1. Reported ROC-AUC values on this dataset cluster between about 0.63 and 0.69 when the target is readmission within 30 days and the whole population is used. Values far above this range come with caveats: age subgroups [18], accuracy-only reporting [21], or details we could not verify (Section 5).
2. Only one of the reviewed studies [14] states that it keeps the encounters of one patient in the same fold. None applies patient grouping inside a stacking procedure.
3. None of the reviewed studies reports calibration (Brier score, ECE, reliability curves).
4. None reports a stacked ensemble that combines XGBoost, LightGBM and CatBoost with out-of-fold predictions. Ensemble averaging appears in [18], but with a random split.
5. Our results add an honest negative finding. Under a leakage-free protocol, stacking the three boosted models gives no meaningful gain over LightGBM alone or a simple average, because the base predictions are highly correlated (0.95–0.98; `results/base_oof_correlation_seed42.csv`).

## 4. Background and methods sources

| Topic | Source | Link |
|---|---|---|
| Readmission burden, Medicare | Jencks et al., NEJM 2009 [1] | https://doi.org/10.1056/NEJMsa0803563 |
| Readmission penalties (HRRP) | McIlvennan et al., Circulation 2015 [2] | https://doi.org/10.1161/CIRCULATIONAHA.114.010270 |
| Readmission in diabetes | Rubin, Curr. Diab. Rep. 2015 [3] | https://doi.org/10.1007/s11892-015-0584-7 |
| Original dataset study | Strack et al., BioMed Res. Int. 2014 [4] | https://doi.org/10.1155/2014/781670 |
| Systematic review of readmission models | Kansagara et al., JAMA 2011 [6] | https://doi.org/10.1001/jama.2011.1515 |
| XGBoost | Chen & Guestrin, KDD 2016 [7] | https://doi.org/10.1145/2939672.2939785 |
| LightGBM | Ke et al., NeurIPS 2017 [8] | https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html |
| CatBoost | Prokhorenkova et al., NeurIPS 2018 [9] | https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html |
| Stacked generalization | Wolpert, Neural Netw. 1992 [10] | https://doi.org/10.1016/S0893-6080(05)80023-1 |
| Data leakage in ML science | Kapoor & Narayanan, Patterns 2023 [11] | https://doi.org/10.1016/j.patter.2023.100804 |
| scikit-learn | Pedregosa et al., JMLR 2011 [12] | https://jmlr.org/papers/v12/pedregosa11a.html |
| Calibration in clinical prediction | Van Calster et al., BMC Med. 2019 [13] | https://doi.org/10.1186/s12916-019-1466-7 |
| SHAP | Lundberg & Lee, NeurIPS 2017 [15] | https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html |
| TreeSHAP | Lundberg et al., Nat. Mach. Intell. 2020 [16] | https://doi.org/10.1038/s42256-019-0138-9 |
| Platt scaling | Platt, 1999 [22] | (book chapter, MIT Press; no DOI) |
| Isotonic calibration | Zadrozny & Elkan, KDD 2002 [23] | https://doi.org/10.1145/775047.775151 |
| Calibration of boosted models | Niculescu-Mizil & Caruana, ICML 2005 [24] | https://doi.org/10.1145/1102351.1102430 |
| Optuna | Akiba et al., KDD 2019 [25] | https://doi.org/10.1145/3292500.3330701 |
| TPE sampler | Bergstra et al., NeurIPS 2011 [26] | https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html |
| Bootstrap | Efron & Tibshirani, 1993 [27] | (book, Chapman & Hall) |
| Reporting guideline | Collins et al., TRIPOD+AI, BMJ 2024 [28] | https://doi.org/10.1136/bmj-2023-078378 |

Links to the studies in Section 2:

| Study | Link |
|---|---|
| Strack et al. 2014 [4] | https://doi.org/10.1155/2014/781670 (full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/) |
| Bhuvan et al. 2016 [17] | https://arxiv.org/abs/1602.04257 |
| Mingle 2017 [18] | https://doi.org/10.19080/CTBEB.2017.07.555715 (full text: https://juniperpublishers.com/ctbeb/CTBEB.MS.ID.555715.php) |
| Shang et al. 2021 [19] | https://doi.org/10.1186/s12911-021-01423-y (full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC8323261/) |
| Liu, Sue & Wu 2024 [14] | https://doi.org/10.21037/jmai-24-70 (full text: https://jmai.amegroups.org/article/view/9179/html) |
| Zarghani 2024 [21] | https://arxiv.org/abs/2406.19980 |
| Emi-Johnson & Nkrumah 2025 [20] | https://doi.org/10.7759/cureus.82437 (full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/) |

## 5. Sources found but not used in the paper

| Source | Why not used |
|---|---|
| Hammoudeh et al., "Predicting hospital readmission among diabetics using deep learning," Procedia Comput. Sci., vol. 141, pp. 484–489, 2018, https://doi.org/10.1016/j.procs.2018.10.138 | A search summary attributes an AUC of 0.79 to this paper, but the full text could not be opened (publisher returned 403), so the outcome definition, split and score could not be checked. Read it via your library before citing. |
| "A Machine Learning Approach for Predicting 30-Day Hospital Readmission in Patients with Diabetes," Healthcare (MDPI), 2026, 14(9):1185, https://www.mdpi.com/2227-9032/14/9/1185 | Appeared in a search listing but was not opened. Check it; it may be relevant recent work. |
| GitHub project "diabetes-readmission-prediction" with patient-level validation, https://github.com/Dongwoon1d/diabetes-readmission-prediction | Not peer reviewed. |

## 6. Dataset and tool links

| Item | Link |
|---|---|
| UCI dataset page (Diabetes 130-US Hospitals 1999-2008, CC BY 4.0) | https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008 |
| UCI dataset DOI [5] | https://doi.org/10.24432/C5230J |
| Direct download (zip) | https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip |
| Our Kaggle copy (private) | https://www.kaggle.com/datasets/vedeshp/diabetes-130-us-hospitals-uci-296 |
| Our Kaggle pipeline notebook (private) | https://www.kaggle.com/code/vedeshp/readmission-full |
| Project code | https://github.com/VedeshP/stacked-gbdt-diabetes-readmission |
| Fairlearn loader for the same dataset (documentation) | https://fairlearn.org/v0.9/user_guide/datasets/diabetes_hospital_data.html |
| TRIPOD+AI checklist (full text) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11025451/ |
| scikit-learn StratifiedGroupKFold | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html |
| XGBoost docs | https://xgboost.readthedocs.io/ |
| LightGBM docs | https://lightgbm.readthedocs.io/ |
| CatBoost docs | https://catboost.ai/docs/ |
| SHAP docs | https://shap.readthedocs.io/ |
| Optuna docs | https://optuna.readthedocs.io/ |
