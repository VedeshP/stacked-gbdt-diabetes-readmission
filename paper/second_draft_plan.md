# Plan for the Second Draft

Status: first draft of Sections I–V exists in `paper/draft.md` (title, abstract, I–V). Results (VI) and Conclusion (VII) are not written yet. This file collects the rules agreed so far, the figures and tables we need, and every change to make for draft 2. Tick items off as they are done.

## A. Rules for the paper (keep adding here)

| # | Rule | Source | Status in draft 1 |
|---|---|---|---|
| R1 | 6-page IEEE conference paper; pages 1–5 content, page 6 mostly references | Professor | Text budget ~3,300–3,600 words incl. results |
| R2 | Title in title case, single part, no subtitle (IEEE Xplore drops subtitles) | IEEE template, style manual | Done: "Patient-Level Evaluation of Gradient-Boosted Ensembles for Diabetic Readmission Prediction" |
| R3 | **Citations only in Section I (Introduction) and Section II (Related Work / literature). No citations in Sections III onward.** | Professor | **Violated**: 13 citations in III–V (see Section D) |
| R4 | Every figure and every table must be cited in the text, in numerical order ("Fig. 1", "Table I") | IEEE style manual | Partly: current mentions point to files, not final figures |
| R5 | The proposed methodology must appear as a figure (draw.io / Canva / Excalidraw) | Professor | **Missing** |
| R6 | References must include papers from 2025 and 2026 | Professor | Only 1 paper from 2025 ([20]); none from 2026 |
| R7 | Plagiarism must be checked; the reference list is excluded from the check | Professor | Not done |
| R8 | All numbers come from `results/`; no invented numbers; TODO when missing | CLAUDE.md | Followed |
| R9 | Report modest/negative results honestly (stacking ≈ single model) | CLAUDE.md | Followed |
| R10 | AI-use disclosure in the Acknowledgment (system name, sections, level of use) | IEEE AI policy | **Missing** |
| R11 | Index Terms in alphabetical order; abstract 150–250 words, one paragraph, no citations | IEEE style manual | Done |
| R12 | Equations built in Word's equation editor, numbered (1), (2) flush right | IEEE template | To do in Word |

## B. Figures

Page budget: about four figures and three tables in five pages. Captions go **below** figures, and every figure must be cited in the text **before** it appears. Export at 300 dpi or higher (or as vector PDF/EMF), sized to one column (about 8.9 cm) unless marked double-column. Check that the text is readable at that size and that the figure still works in grayscale.

| Fig. | Content | Status | Source file | Where it is cited |
|---|---|---|---|---|
| 1 | **Proposed methodology (pipeline diagram)** — required (R5) | **To draw** | draw.io / Canva / Excalidraw → `paper/figures/fig1_methodology.png` | Start of Section IV ("The proposed pipeline is shown in Fig. 1.") |
| 2 | ROC and precision-recall curves on the test set (XGBoost, LightGBM, CatBoost, average, stack) | Exists | `results/roc_pr_curves_seed42.png` (double-column, or one column if shrunk) | Section VI-A, model comparison |
| 3 | Reliability diagram: best single model raw, stack raw, stack + Platt, stack + isotonic | Exists | `results/reliability_seed42.png` | Section VI-C, calibration |
| 4 | SHAP: top features of the stacked model (mean absolute attribution) | Exists | `results/shap_top_features_stack.png` | Section VI-D, explainability |
| (opt.) | 30-day readmission rate by prior inpatient visits (with 95% CIs) | Exists | `results/eda_readmit_rate_by_number_inpatient.png` | Section III, only if space allows |
| (opt.) | SHAP beeswarm of the best single model | Exists | `results/shap_summary_lgbm.png` | Only if space allows |

Draft 1 currently mentions "Fig. 1, see results/eda_missingness.png" in Section III-B. In draft 2, **remove that mention** (missingness is covered in the text) so that Fig. 1 is the methodology figure.

### Fig. 1 content (what to draw)

Draw it left to right as two lanes, a training lane and a test lane, with a dashed box around everything that uses only training patients. Keep the labels short; the numbers come from `results/split.json`.

1. **Data**: UCI Diabetes 130-US Hospitals, 101,766 encounters, 71,518 patients.
2. **Cohort and features**: exclude death/hospice (2,423); ICD-9 → 9 groups; medication counts; 36 features.
3. **Patient-level split** (StratifiedGroupKFold on patient ID): Training 55,952 patients / 79,625 encounters; Test 14,038 patients / 19,718 encounters. No shared patients.
4. **Training lane (patient-grouped 5-fold CV inside)**:
   - Optuna tuning (PR-AUC, grouped folds) → tuned XGBoost, LightGBM, CatBoost (plus LR and RF as references);
   - out-of-fold probabilities from each base model → logit → **logistic-regression meta-learner** (Eq. 1); a parallel branch for the **simple average**;
   - calibration (none / Platt / isotonic) and thresholds (max-F1, recall 0.70), fitted on out-of-fold predictions.
5. **Test lane (used once)**: refitted models → probabilities → metrics (ROC-AUC, PR-AUC, Brier, ECE, precision, recall, F1) with patient-level bootstrap 95% CIs.
6. **Explanation**: TreeSHAP per base model, weighted by meta-learner weights → exact stack attributions (Eq. 2).

Tips: use the same color for "training-only" boxes and a different one for "test" boxes; label arrows with what flows (e.g. "OOF probabilities"); avoid more than about 15 boxes; export at double-column width if it has to be wide. Claude can generate a draw.io (`.drawio`) starting file for you to edit; ask if you want that.

## C. Tables

Captions go **above** tables ("TABLE I" centered, title below it), and each table must be cited in the text.

| Table | Content | Source | Cited in |
|---|---|---|---|
| I | Dataset and split summary (encounters, patients, positives and rate: full / after exclusion / train / test) | `results/eda.json`, `results/split.json` | Section III |
| II | Main comparison on the test set: LR, RF, XGBoost, LightGBM, CatBoost, average, stack — ROC-AUC, PR-AUC, Brier, ECE, F1 (mean ± std over seeds; 95% CI) | `results/baselines_test.csv`, `results/stacking_comparison.csv`, `results/stacking_test_ci_seed42.csv` | Section VI-A |
| III | Ablations: class weights vs none, and calibration none / Platt / isotonic (Brier, ECE, ROC-AUC, PR-AUC) | `results/imbalance_ablation.csv`, `results/calibration_comparison.csv` | Section VI-B/C |
| (drop or move) | Tuned hyperparameters | `results/best_params.json` | Draft 1 calls this "Table III"; in draft 2, drop it (say "available in the repository") to save space |

## D. Citations to move out of Sections III–V (rule R3)

Every citation below must be removed from its current place. Where the source still matters, introduce it in Section I or II instead, so the reference stays in the list.

| Ref | Currently in | What it supports | Plan |
|---|---|---|---|
| [5] dataset | III-A "The dataset [5]" | Dataset source | Already cited in I; in III write "The dataset described in Section I…" without a number |
| [4] Strack et al. | III-B (×2) | Test-ordering informative; ICD-9 grouping | Already cited in I and II; in II add one sentence that [4] grouped ICD-9 codes into clinical categories and found that whether HbA1c was measured matters; in III refer to "the grouping of Strack et al." without a number, or just describe it |
| [12] scikit-learn | IV-A | Pipelines | Already cited in I; drop the bracket in IV |
| [9] CatBoost | IV-A | Native categorical handling | Already described in II; drop the bracket in IV |
| [10] stacking | IV-B | Stacked generalization | Already in I and II; drop the bracket |
| [22] Platt, [23] isotonic | IV-C | Calibration methods | Already in II; drop the brackets |
| [15], [16] SHAP, TreeSHAP | IV-D | SHAP | Already in I and II; drop the brackets |
| [25] Optuna, [26] TPE | V | Tuning tools | **Only cited here.** Add a sentence in II (e.g. on tuning practice: "Bayesian optimization frameworks such as Optuna [25] with the TPE sampler [26]…") or drop both references |
| [27] bootstrap | V | Cluster bootstrap CIs | **Only cited here.** Mention in II together with calibration/evaluation practice, or drop |
| [28] TRIPOD+AI | V | Reporting guideline | **Only cited here.** Move to I (e.g. after the calibration argument: "reporting guidelines such as TRIPOD+AI [28] ask for…") |

After moving them, re-run the renumbering script (order of first citation) and check that every reference is still cited at least once.

## E. References from 2025 and 2026 (rule R6)

Goal: at least 4–6 recent references (2025–2026), each verified (open the paper, check the venue, year and the claim we cite it for).

Current: [20] Emi-Johnson & Nkrumah, Cureus, 2025.

Leads to check (found during the literature search; **not yet verified**):
- "A Machine Learning Approach for Predicting 30-Day Hospital Readmission in Patients with Diabetes," Healthcare (MDPI), 2026, vol. 14, no. 9, art. 1185, https://www.mdpi.com/2227-9032/14/9/1185.
- "Machine learning-based prediction model for 30-day readmission risk in elderly patients with type 2 diabetes mellitus and heart failure… with SHAP interpretability analysis," PMC12819643 (year to confirm).
- A 2025–2026 study using patient-level validation or calibration for readmission prediction (search PubMed / Google Scholar: "readmission prediction calibration 2025", "patient-level split leakage EHR 2025").
- A 2025–2026 review of machine learning for hospital readmission, or of data leakage in clinical ML.
- Recent work on stacking/ensembles for tabular clinical data (2025–2026).

For each new reference, record in `paper/literature_review.md`: what it did, the split, the metric, the score, and the link. Cite it in Section I or II only.

## F. Plagiarism check (rule R7)

1. Use the similarity checker your university provides (usually Turnitin or iThenticate; IEEE screens submissions with similar software). Exclude the reference list and quoted material when running the check, as agreed.
2. Run it on the full Word file **after** all edits, because new sentences change the score.
3. Typical risk spots in this draft:
   - Related Work sentences that summarize other papers: paraphrase in your own words; never copy abstract sentences;
   - dataset descriptions, which tend to match the UCI page and other papers almost word for word;
   - standard method descriptions (Platt scaling, isotonic regression, SHAP).
4. Rewrite any highlighted passage instead of only swapping words.
5. Keep the report (PDF) for your records; some conferences ask for it.
6. The AI-use disclosure (R10) is separate from plagiarism; it is required regardless of the similarity score.

## G. Changes to make in draft 2 (checklist)

Content still to write:
- [ ] Section VI Results and Discussion (~800–900 words): Table II, Table III, Fig. 2–4; stacking ≈ single model, explained by the 0.95–0.98 correlation; calibration already good, class weights hurt Brier; operating point at 70% recall; SHAP; comparison with prior work (in words, without citations, per R3; refer back to Section II).
- [ ] Section VII Limitations and Conclusion (~250–300 words): 1999–2008 data, one data source, no external validation, CatBoost tuning budget, not for clinical use; future work.
- [ ] Acknowledgment with the AI-use disclosure (R10).

Edits to existing text:
- [ ] Insert Fig. 1 (methodology) and cite it at the start of Section IV (R4, R5).
- [ ] Remove all citations from Sections III–V as in Section D (R3).
- [ ] Remove "Fig. 1, see results/eda_missingness.png" from III-B; keep Table I (dataset/split) and cite it in III.
- [ ] Drop "Table III (see results/best_params.json)" in V or replace with "listed in the repository".
- [ ] Add 2025–2026 references to Sections I/II (R6) and re-number.
- [ ] Fill TODOs: repository URL and commit hash; CPU core count and run time (needs a new Kaggle run, which now records the core count); sensitivity analysis with death/hospice encounters included (`EXCLUDE_EXPIRED=0`) or drop that sentence.
- [ ] Decide whether to re-run with a larger CatBoost tuning budget (only 4 completed trials) before final numbers.
- [ ] Re-check total length in the Word template: pages 1–5 content, page 6 references (R1).
- [ ] Resolve TODO-VERIFY items in `paper/references.md` (page ranges, issue numbers, co-author lists).
- [ ] Build equations (1) and (2) in Word's equation editor (R12).
- [ ] Run the plagiarism check (R7) on the final Word file.
- [ ] Final pass with the checklist in `paper/ieee_guide.md` Section 8.

## H. Open questions for the author

- Target conference and its exact page limit, paper size (A4 or Letter) and review type (blind or not)?
- Should LR and RF stay in Table II as references, or only the boosted models?
- Keep the optional EDA figure if space allows?
