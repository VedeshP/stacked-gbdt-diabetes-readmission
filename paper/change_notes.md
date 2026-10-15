# Change Notes for `paper/draft.md` (2026-10-15)

Use this to plan your rewriting. **No further content changes are planned from my side**, so every paragraph is safe to rewrite now, in parallel. The tables below show what changed today compared with the previous version (commit `2645d3b`), and what you must keep when you rewrite.

## 1. What changed today

1. **Author block:** email added under the name.
2. **Abbreviations:** spelled out at first use, as the IEEE style manual requires.
   - Abstract: ROC-AUC, PR-AUC, CI and SHAP written out in words. It was also shortened to 248 words.
   - Body: SHAP (Section I); ICD-9, HbA1c, PR-AUC, ROC-AUC and SMOTE (Section II); CI (Section V).
3. **Figures and tables placed:** each one has an "[Insert … here: file]" line and an IEEE caption at the place it belongs.
   - Table I (III-A)
   - Fig. 1 (IV, top)
   - Table II and Fig. 2 (VI-A)
   - Table III (VI-B)
   - Fig. 3 (VI-C)
   - Fig. 4 (VI-E)

   Table captions sit **above** the table and figure captions **below** the figure.
4. **"(see results/…)" removed from the running text** where a table or figure now carries the information. File names appear only in the "[Insert …]" lines.
5. **Fig. 4 redrawn** with readable feature names; patient-history features are marked with *.
6. **Final figures** collected in `paper/figures/` (fig1–fig4); tables are in `results/table1_cohort.csv`, `table2_main_results.csv` and `table3_ablations.csv`.

## 2. Paragraph map

UNCHANGED = same text as the previous version. EDITED = small wording change today (abbreviations or figure references). NEW = added today (figure/table placement lines and captions).

| Section | Paragraph (first words) | Status today |
|---|---|---|
| Front | Title | UNCHANGED |
| Front | Author block | EDITED (email added) |
| Front | Abstract | EDITED (abbreviations spelled out, shortened) |
| Front | Index Terms | UNCHANGED |
| I | "Readmission shortly after discharge…" | UNCHANGED |
| I | "Patients with diabetes are an important group…" | EDITED (SHAP spelled out) |
| I | "Three methodological issues…" | UNCHANGED |
| I | "A further question is what information…" | UNCHANGED |
| I | "We revisit the task…" + contributions 1)–4) | UNCHANGED |
| II | "Kansagara et al. …" | EDITED (ICD-9, HbA1c, PR-AUC, ROC-AUC, SMOTE spelled out) |
| II | "Ensembles have also been studied…" | UNCHANGED |
| II | "The three boosted learners differ…" | UNCHANGED |
| III-A | "The Diabetes 130-US Hospitals dataset…" | EDITED (last sentence) |
| III-A | Table I placement + caption | NEW |
| III-B | "Weight (96.9% missing)…" | UNCHANGED |
| III-C | "The dataset does not contain dates…" | UNCHANGED |
| III-D | "Patients have up to 40 encounters…" | UNCHANGED |
| IV | "The complete pipeline is shown in Fig. 1…" | EDITED |
| IV | Fig. 1 placement + caption | NEW |
| IV-A | "We use L2-regularized logistic regression…" | UNCHANGED |
| IV-B | "The training set is split…", Eq. (1), "where sigma…", "Thresholds and calibrators…" | UNCHANGED |
| IV-C | "About 11% of training encounters…" | UNCHANGED |
| IV-D | "We compute exact TreeSHAP…", Eq. (2), "so sum_m…" | UNCHANGED |
| V | Metrics, Tuning, Implementation | UNCHANGED |
| V | Protocol | EDITED (CI spelled out) |
| VI-A | "Table II reports test performance…" | EDITED |
| VI-A | Table II placement + caption; Fig. 2 placement + caption | NEW |
| VI-A | "The gain from ensembling…", "Our test ROC-AUC values…" | UNCHANGED |
| VI-B | "The feature ablation in Table III…" | EDITED |
| VI-B | Table III placement + caption | NEW |
| VI-C | "Class weighting does not help…" | EDITED (figure reference only) |
| VI-C | Fig. 3 placement + caption | NEW |
| VI-D | "At the threshold that maximizes F1…" | UNCHANGED |
| VI-E | "Fig. 4 shows…" | EDITED (figure reference only) |
| VI-E | Fig. 4 placement + caption | NEW |
| VII | Both paragraphs | UNCHANGED |
| Acknowledgment | AI-use disclosure | UNCHANGED |

## 3. What to keep exactly when you rewrite

These items are checked against `results/` and the rules. Reword the sentences around them freely, but do not change the items themselves:

- **Every number.** Each one comes from `results/paper_numbers.json` or a results file. If a sentence's number moves, keep the same value and rounding.
- **Citation numbers and where they appear.** Citations stay in Sections I and II only. If you reorder sentences in I or II so that a reference is cited for the first time in a new position, the numbering must be redone (ask me, or renumber by order of first citation).
- **Figure and table references** ("Fig. 2", "Table III"): each must be mentioned in the text before it appears, in numerical order.
- **Equations (1) and (2)** and the symbol definitions right after them.
- **Abbreviation definitions at first use**, in the abstract and in the body.
- **Claims tied to verified sources** in Section II, e.g. "they note that their cross-validation did not enforce patient-level splits" for Salim and Ibrahim. Paraphrase, but do not change the meaning.
- **The abstract word limit** (150–250 words, now 248) and **Index Terms in alphabetical order**.
- **The Acknowledgment's AI-use disclosure.** IEEE policy requires it; you can reword it, as long as it still names the tool, the sections and the level of use.
- **The honesty points:** stacking adds only about 0.002; the patient-level protocol; retrospective, not for clinical use.

## 4. Compiled guidance and how the draft meets it

| # | Rule (source) | Status in the draft |
|---|---|---|
| R1 | 6 pages: pages 1–5 content, page 6 references (professor) | About 3,700 words of text + 4 figures + 3 tables. Check in the Word template; if long, trim Section II or IV first |
| R2 | Title case, single-part title, no subtitle (IEEE) | Done |
| R3 | Citations only in Sections I and II (professor) | Done: 0 citations in III–VII |
| R4 | Every figure/table cited in text, in order (IEEE) | Done: Table I → Fig. 1 → Table II → Fig. 2 → Table III → Fig. 3 → Fig. 4 |
| R5 | Methodology as a figure (professor) | Done: Fig. 1; to redraw it yourself, see `paper/flowchart_guide.md` |
| R6 | References from 2025 and 2026 (professor) | Done: five ([11], [12], [13], [15], [25]) |
| R7 | Plagiarism check, references excluded (professor) | **Your step**, after rewriting |
| R8 | All numbers from `results/` (CLAUDE.md) | Done; see `results/paper_numbers.json` |
| R9 | Honest reporting (CLAUDE.md) | Done |
| R10 | AI-use disclosure in Acknowledgment (IEEE) | Done |
| R11 | Abstract 150–250 words, Index Terms alphabetical, abbreviations defined (IEEE) | Done |
| R12 | Equations in Word's equation editor, numbered (IEEE template) | **Your step** in Word (UnicodeMath strings given earlier in chat) |
| — | Captions: table above, figure below; "Fig." in text (IEEE) | Done in the draft; apply caption styles in Word |
| — | No TODOs left (author) | Done |

## 5. Still to do in Word (not in the draft text)

1. Add your **affiliation** (department, university, city, country) under your name; the draft has only the email.
2. Build Eq. (1) and Eq. (2) with the equation editor.
3. Insert the 4 figures and 3 tables at the "[Insert …]" lines, then delete those lines. Fig. 1 and Fig. 2 are double-column (18 cm); Figs. 3–4 single-column (8.9 cm).
4. Paste `paper/references.md` under REFERENCES without the verification notes.
5. Check the page count (R1), then run the plagiarism check (R7).
