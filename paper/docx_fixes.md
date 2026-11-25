# Fixes for diabetes_readmission.docx (checked 2026-10-15)

The Word file was checked paragraph by paragraph against `paper/draft.md` (latest version). Most of the text is current, but several paragraphs are **older versions** with wrong citation numbers, some paragraphs are **missing**, two are **duplicated**, and the **reference list is the old one**. Below: what to fix, in document order, with the exact text to paste. Figures and tables are not listed here, because you will add them later.

## A. Front matter

1. **Title** uses the "Author" style. Apply the template's **"Title"** style ("paper title").
2. **Author blocks still contain template text.**
   - Vedesh Pandya: "line 2: dept. name of organization (of Affiliation) / line 3: name of organization (of Affiliation) / line 4: City, Country". Replace it with your department, university, city and country.
   - Chetan Gadhiya: all lines are template text, including "line 5: email address or ORCID". Fill them in.
   - Third author block ("line 1: 3rd Given Name Surname …"): delete it if there is no third author, along with any other unused template author blocks.
3. **Footer** has the template placeholder "XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE". Keep or replace it as your conference instructs. Usually the conference supplies the real copyright line for the final version; for submission it is often removed.
4. Abstract: current (248 words). OK.
5. "Keywords—…" instead of "Index Terms—…": fine; that is the conference template's wording.

## B. Section I (Introduction)

6. **Paragraph "Three methodological issues…" is an OLD version with wrong citations.** It cites [11], [12], [13] for leakage, scikit-learn and calibration, but [11]–[13] are now Salim, Wang and Afridi. Replace the whole paragraph with:

Three methodological issues limit how far published results on this dataset can be trusted and compared. First, the data record encounters, not patients: 46.2% of encounters come from patients admitted more than once. A random split of encounters places the same patient in both training and test sets, a form of leakage known to inflate reported performance [14], [15]. In stacked ensembles the issue recurs, because the out-of-fold predictions that train the meta-learner must also be patient-grouped, and the stacking estimator of scikit-learn [16] (version 1.7) does not accept group labels. Second, calibration is rarely reported, although a risk score used to allocate resources must give probabilities that match observed rates [17], and reporting guidelines ask for it explicitly [18]. Third, explanations of a single ensemble member do not describe what the ensemble as a whole relies on.

7. **Missing paragraph.** Insert after the paragraph above, before "We revisit the task…":

A further question is what information the models receive. Prior use of hospital services is among the most common predictors in readmission models [6], yet studies on this dataset treat every encounter as independent and ignore the patient's own earlier stays, which are known at a later discharge.

## C. Section II (Related Work)

8. **Missing paragraph.** Insert after the "Kansagara et al. [6]…" paragraph. It contains the 2025–2026 references and the closest prior work:

Ensembles have also been studied. Lu and Uddin [27] built an explainable stacking model with random under-sampling for this task. Salim and Ibrahim [11] combined logistic regression, random forest, XGBoost and LightGBM in a stacking ensemble with Platt scaling and reported a nested cross-validated ROC-AUC of 0.664, a calibrated ROC-AUC of 0.688 and a Brier score of 0.094; they note that their cross-validation did not enforce patient-level splits. Afridi et al. [13] proposed a leakage-resistant framework in which resampling is restricted to the training folds. In other clinical domains, patient-level and temporal leakage has been shown to inflate published results [15]. To our knowledge, no study on this dataset evaluates a stacked ensemble with patient-grouped out-of-fold predictions while also using the patient's earlier encounters as features.

9. **Paragraph "The three boosted learners differ in design…" is an OLD version with wrong citation numbers** ([22], [23], [24], [15], [16]), and it lacks Optuna [28], TPE [29] and the bootstrap [33]. Replace it with:

The three boosted learners differ in design: XGBoost [7] grows regularized trees level by level, LightGBM [8] grows trees leaf-wise on histogram bins, and CatBoost [9] builds symmetric trees and encodes categories with ordered target statistics. Such differences can produce partly different errors, the condition under which stacking [10] helps. Their hyperparameters are commonly tuned with Optuna [28] and the tree-structured Parzen estimator [29]. Platt scaling [30] and isotonic regression [31] are standard post-hoc calibration methods, and boosted models can produce distorted probabilities that benefit from them [32]. The uncertainty of performance estimates can be quantified by bootstrap resampling [33], which must respect clustering when patients contribute several encounters. SHAP [20] with exact tree algorithms [21] is the most common way to explain boosted models.

## D. Section III (Dataset and Preprocessing)

10. Heading "dataset and preprocessing" is typed in lowercase. Type it as "Dataset and Preprocessing"; the heading style turns it into small caps.
11. **"Features" paragraph is an OLD version.** It refers to the wrong figure ("Fig. 1, see results/eda_missingness.png"), cites [4] twice (citations are not allowed in Section III), gives the old feature count (36/14/22 instead of 44/17/27) and does not mention 3-digit codes. Rename the subsection to **"Encounter Features"** and replace the paragraph with:

Weight (96.9% missing) is dropped; medical specialty (49.1% missing) and payer code (39.6% missing) keep missingness as a category. For HbA1c and glucose, the value "None" means the test was not performed (83.3% and 94.7% of encounters) and is kept as a category, because ordering the test is itself informative. Each ICD-9 diagnosis code is mapped to one of nine clinical groups (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms and other) and is also kept as a three-character code for finer detail. Fifteen near-constant medication columns are removed, and counts of active medications and dosage changes are added, together with age-band midpoints and a total of prior-year visits.

12. **Missing subsection "Patient-History Features".** This is the main addition that raised the ROC-AUC, and the paper's contribution 2 depends on it. Insert a Heading 2 "Patient-History Features" after "Encounter Features", with this paragraph:

The dataset does not contain dates, so we use the encounter identifier as the time order within each patient. This order is consistent with the labels: of the 30,248 encounters that have an earlier encounter of the same patient, only 1.6% follow an encounter that was labeled as not readmitted (see results/encounter_order_check.json). For each encounter we compute, from the same patient's earlier encounters only, the number of earlier encounters, the number of earlier 30-day readmissions, the outcome label and length of stay of the previous encounter, and its discharge disposition. These quantities are known at the current discharge; no later information is used, and the history is computed before any exclusion. The final feature set has 44 features, 17 numeric and 27 categorical (see results/features.csv). All steps in this section are row-wise; fitted transformations are learned inside the model pipelines (Section IV).

13. **"Patient-Level Split" paragraph is an OLD version.** It says "(Table II, see results/split.json)", but Table II is now the results table, and the split belongs to Table I. Replace it with:

Patients have up to 40 encounters; 23.5% have more than one, accounting for 46.2% of encounters. We split by patient with stratified group 5-fold partitioning (seed 42) and hold out one fold as the test set: 79,625 training encounters from 55,952 patients (positive rate 11.46%) and 19,718 test encounters from 14,038 patients (11.10%), with no patient in both. All later steps use patient-grouped folds of the training set, and the test set is used once.

## E. Section IV (Methodology)

14. Equations (1) and (2) are built with the equation editor. Check that the numbers (1) and (2) show flush right. The text extract shows "#1" and "#2"; if Word displays "#1" literally, press Enter at the end of each equation, or retype the number as described earlier.
15. Paragraph after Eq. (2): its start ("so sum_m w_m phi_ijm …") was converted into an equation object. Make sure it reads as normal text with an inline symbol: "so Σ_m w_m φ_ijm is an exact attribution …".
16. In the text, write the symbols as math too (σ instead of "sigma", p_im, w_m, φ_ijm as italic with subscripts), so they match the equations. Optional, but reviewers notice it.

## F. Section V (Experimental Setup)

17. **"Implementation" is duplicated.** Delete the FIRST "Implementation" paragraph, which ends with "(see results/env_kaggle.json; core count and run time TODO). Reporting follows TRIPOD+AI where applicable [28]." It contains a TODO and a citation, and citations are not allowed in Section V. Keep the second one:

Implementation. Python 3.13 with scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, Optuna 5.0.0 and SHAP 0.52.0, run as one Kaggle notebook on 4 CPU cores without a GPU; the complete pipeline took 251 minutes (see results/env_kaggle.json and results/runtime_kaggle.json). The code, the notebook builder and all result files are available at https://github.com/VedeshP/stacked-gbdt-diabetes-readmission (commit c4a8ddf).

18. Optional for the final version: the text mentions repository paths ("results/best_params.json", "results/shap_additivity_check.json", "results/env_kaggle.json"). You can replace them with "(available in the code repository)".

## G. Section VI (Results and Discussion)

19. **"Effect of Features" paragraph is duplicated.** Delete one of the two identical paragraphs that start "The feature ablation in Table III…".
20. Everything else in Section VI is current. Tables II–III and Figs. 2–4 will go in when you add the figures.

## H. Section VII and after

21. Section VII: current. OK.
22. **ACKNOWLEDGMENT is missing.** It is required by IEEE policy because AI was used. Add the heading "Acknowledgment" (template style, not numbered) before References, with:

The author used Claude (Anthropic, model Claude Opus 5.5) to help write and debug the analysis code, to generate Fig. 1, to search for and check literature, and to draft the text of all sections. The author reviewed, verified and edited all AI-assisted content and takes full responsibility for the paper.

With two authors, change "The author used … The author reviewed …" to "The authors used … The authors reviewed … and take full responsibility for the paper."

## I. References (replace the whole list)

23. **The reference list in the Word file is the old 28-entry version.**
   - Entry [2] was replaced by the template example "J. Clerk Maxwell, A Treatise on Electricity and Magnetism…".
   - The five new references (Salim, Wang, Afridi, Zablah, Lu & Uddin) are missing.
   - The order no longer matches the citation numbers in the text.
   - LightGBM and TPE still show unverified page ranges.

   Delete all entries and paste this list (33 entries). Then apply the template's "references" style, which adds the [n] numbers automatically, so remove the typed "[n]" at the start of each line if the style already numbers them:

[1] S. F. Jencks, M. V. Williams, and E. A. Coleman, "Rehospitalizations among patients in the Medicare fee-for-service program," N. Engl. J. Med., vol. 360, no. 14, pp. 1418–1428, 2009, doi: 10.1056/NEJMsa0803563.

[2] C. K. McIlvennan, Z. J. Eapen, and L. A. Allen, "Hospital readmissions reduction program," Circulation, vol. 131, no. 20, pp. 1796–1803, 2015, doi: 10.1161/CIRCULATIONAHA.114.010270.

[3] D. J. Rubin, "Hospital readmission of patients with diabetes," Curr. Diab. Rep., vol. 15, no. 4, Art. no. 17, 2015, doi: 10.1007/s11892-015-0584-7.

[4] B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios, and J. N. Clore, "Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records," BioMed Res. Int., vol. 2014, Art. no. 781670, 2014, doi: 10.1155/2014/781670.

[5] J. Clore, K. Cios, J. DeShazo, and B. Strack, "Diabetes 130-US Hospitals for Years 1999-2008," UCI Machine Learning Repository, 2014, doi: 10.24432/C5230J.

[6] D. Kansagara, H. Englander, A. Salanitro, D. Kagen, C. Theobald, M. Freeman, and S. Kripalani, "Risk prediction models for hospital readmission: A systematic review," JAMA, vol. 306, no. 15, pp. 1688–1698, 2011, doi: 10.1001/jama.2011.1515.

[7] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[8] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in Adv. Neural Inf. Process. Syst., vol. 30, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

[9] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in Adv. Neural Inf. Process. Syst., vol. 31, 2018, pp. 6638–6648. [Online]. Available: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

[10] D. H. Wolpert, "Stacked generalization," Neural Netw., vol. 5, no. 2, pp. 241–259, 1992, doi: 10.1016/S0893-6080(05)80023-1.

[11] S. S. Salim and A. A. Ibrahim, "A machine learning approach for predicting 30-day hospital readmission in patients with diabetes," Healthcare, vol. 14, no. 9, Art. no. 1185, 2026, doi: 10.3390/healthcare14091185.

[12] L. Wang, C. Wei, Y. Chen, D.-S. Lu, H.-X. Zhang, and M. Yang, "Machine learning-based prediction model for 30-day readmission risk in elderly patients with type 2 diabetes mellitus and heart failure: A retrospective cohort study with SHAP interpretability analysis," Front. Cardiovasc. Med., vol. 12, Art. no. 1673159, 2026, doi: 10.3389/fcvm.2025.1673159.

[13] A. A. Afridi, M. Usman, D. J. Rubin, and M. R. R. Rana, "A robust and explainable framework for predicting hospital readmission in patients with diabetes," Diabetology, vol. 7, no. 10, Art. no. 197, 2026, doi: 10.3390/diabetology7100197.

[14] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns, vol. 4, no. 9, Art. no. 100804, 2023, doi: 10.1016/j.patter.2023.100804.

[15] I. Zablah, Y. Molina, and A. Garcia Loureiro, "Patient-level and temporal data leakage can inflate performance in medication recommendation models: A critical appraisal of the original KATMed publication," J. Biomed. Inform., vol. 178, Art. no. 105016, 2026, doi: 10.1016/j.jbi.2026.105016.

[16] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html

[17] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, E. W. Steyerberg, P. Bossuyt, G. S. Collins, P. Macaskill, K. G. M. Moons, and A. J. Vickers, "Calibration: The Achilles heel of predictive analytics," BMC Med., vol. 17, Art. no. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[18] G. S. Collins et al., "TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, Art. no. e078378, 2024, doi: 10.1136/bmj-2023-078378.

[19] V. B. Liu, L. Y. Sue, and Y. Wu, "Comparison of machine learning models for predicting 30-day readmission rates for patients with diabetes," J. Med. Artif. Intell., vol. 7, Art. no. 23, 2024, doi: 10.21037/jmai-24-70.

[20] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 4765–4774. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

[21] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," Nat. Mach. Intell., vol. 2, no. 1, pp. 56–67, 2020, doi: 10.1038/s42256-019-0138-9.

[22] M. S. Bhuvan, A. Kumar, A. Zafar, and V. Kishore, "Identifying diabetic patients with high risk of readmission," 2016, arXiv:1602.04257. [Online]. Available: https://arxiv.org/abs/1602.04257

[23] D. Mingle, "Predicting diabetic readmission rates: Moving beyond HbA1c," Curr. Trends Biomed. Eng. Biosci., vol. 7, no. 3, pp. 55–65, 2017, doi: 10.19080/CTBEB.2017.07.555715.

[24] Y. Shang, K. Jiang, L. Wang, Z. Zhang, S. Zhou, Y. Liu, J. Dong, and H. Wu, "The 30-days hospital readmission risk in diabetic patients: Predictive modeling with machine learning classifiers," BMC Med. Inform. Decis. Mak., vol. 21, suppl. 2, Art. no. 57, 2021, doi: 10.1186/s12911-021-01423-y.

[25] O. G. Emi-Johnson and K. J. Nkrumah, "Predicting 30-day hospital readmission in patients with diabetes using machine learning on electronic health record data," Cureus, vol. 17, no. 4, Art. no. e82437, 2025, doi: 10.7759/cureus.82437.

[26] A. Zarghani, "Comparative analysis of LSTM neural networks and traditional machine learning models for predicting diabetes patient readmission," 2024, arXiv:2406.19980. [Online]. Available: https://arxiv.org/abs/2406.19980

[27] H. Lu and S. Uddin, "Explainable stacking-based model for predicting hospital readmission for diabetic patients," Information, vol. 13, no. 9, Art. no. 436, 2022, doi: 10.3390/info13090436.

[28] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2019, pp. 2623–2631, doi: 10.1145/3292500.3330701.

[29] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, "Algorithms for hyper-parameter optimization," in Adv. Neural Inf. Process. Syst., vol. 24, 2011. [Online]. Available: https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html

[30] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in Advances in Large Margin Classifiers, A. J. Smola, P. Bartlett, B. Schölkopf, and D. Schuurmans, Eds. Cambridge, MA, USA: MIT Press, 1999, pp. 61–74.

[31] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in Proc. 8th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2002, pp. 694–699, doi: 10.1145/775047.775151.

[32] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in Proc. 22nd Int. Conf. Mach. Learn. (ICML), 2005, pp. 625–632, doi: 10.1145/1102351.1102430.

[33] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.

## J. Final checks after these fixes

- Citations [1]–[33] appear in order of first use, and only in Sections I and II.
- Figure/table order when you insert them: Table I (III-A), Fig. 1 (IV), Table II and Fig. 2 (VI-A), Table III (VI-B), Fig. 3 (VI-C), Fig. 4 (VI-E). Captions: tables above, figures below.
- No "TODO", "see results/…" or template text left (search the document for "TODO", "line ", "XXX", "Maxwell").
- Page count: pages 1–5 content, page 6 references.
