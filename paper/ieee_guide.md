# IEEE Conference Paper Guide

A practical guide for turning `paper/draft.md` into an IEEE conference paper in Word. Every rule here links to the source it comes from (Section 9). Where the guide gives advice rather than a rule, it says so. **The call for papers of your target conference always wins**: page limit, paper size, blind review and submission system are set by the conference, not by IEEE in general.

## 1. Download the template

- Official page: **IEEE Manuscript Templates for Conference Proceedings**: https://www.ieee.org/conferences/publishing/templates.html
  - Microsoft Word template, **US Letter** or **A4**. Use the paper size your conference asks for.
  - LaTeX template (ZIP) and LaTeX bibliography files, if you ever switch.
  - Overleaf versions: https://www.overleaf.com/gallery/tagged/ieee-official
- Some conferences publish their own copy of the template. If yours does, use that one.
- Save it in the repo root as `ieee-template.docx` so the pandoc command in `CLAUDE.md` works:
  `pandoc paper/draft.md -o paper/paper.docx --reference-doc=ieee-template.docx`
- **The template contains guidance text (instructions written as sample paragraphs). Delete all of it before submitting.** IEEE warns that papers that keep template text may not be published.

## 2. Getting the draft into Word

`paper/draft.md` is written as plain text so it pastes cleanly:

1. Open the template and keep its styles. Do not change margins, column widths, fonts or spacing; the template sets them.
2. Paste each part and apply the template's styles:
   - Title, then the author block (name, affiliation, city/country, email).
   - "Abstract—…" and "Index Terms—…": the template has dedicated styles for these.
   - Section headings "I. INTRODUCTION" use the Heading 1 style, and subsections "A. Data Source" use Heading 2.
   - Contributions "1) …" are normal paragraphs, or a list in the template's list style.
3. Equations (1) and (2): retype them with Word's equation editor and number them flush right as (1), (2). The draft writes them in plain text (sigma, logit, sum_m) so their meaning is clear.
4. Tables and figures: insert them from `results/` where the draft says "Table I, see results/…" or "Fig. 1, see results/…".
5. References: paste `paper/references.md` (without the verification notes) under the REFERENCES heading and apply the template's reference style.

## 3. Structure of a ~6-page conference paper

Typical layout for this project (adapt to the conference page limit):

| Section | Target length | Content |
|---|---|---|
| Title + authors | — | Specific and informative; avoid unexplained abbreviations in the title |
| Abstract | 150–250 words, one paragraph | Problem, method, main numbers, conclusion; no citations, no equations |
| Index Terms | 4–8 terms, alphabetical | Required by the style manual |
| I. Introduction | ~0.75 page | Motivation, gap, contributions list, paper outline |
| II. Related Work | ~0.5 page | Studies on this dataset with how they split data and what they report |
| III. Dataset and Preprocessing | ~0.75 page | Source, outcome, cohort, features, patient-level split |
| IV. Methodology | ~1 page | Pipelines, grouped OOF stacking, imbalance, calibration, thresholds, SHAP |
| V. Experimental Setup | ~0.5 page | Metrics, tuning, protocol, implementation |
| VI. Results and Discussion | ~1.5 pages | Tables and figures, ablations, calibration, SHAP, comparison with prior work |
| VII. Limitations and Conclusion | ~0.4 page | 1999-2008 data, single source, no external validation, not for clinical use |
| Acknowledgment | short | Funding, help, **AI-use disclosure** (Section 6) |
| References | ~0.5 page | IEEE numbered style |

## 4. Writing conventions (IEEE style)

From the IEEE Editorial Style Manual and the template:

- **Headings:** main sections take Roman numerals (I., II., …) in small caps or upper case as the template styles them; subsections take capital letters (A., B., …). Acknowledgment and References headings are not numbered.
- **Abstract:** one paragraph of 150 to 250 words; no numbered equations, reference citations or footnotes.
- **Index Terms:** required; listed in **alphabetical order** as the last paragraph of the abstract section; capitalize only the first word (and proper nouns or acronyms).
- **Abbreviations:** define each one at first use, in the abstract and again in the body (e.g., "area under the receiver operating characteristic curve (ROC-AUC)"). Standard ones like "IEEE", "SI" or "DNA" need no definition.
- **Figures:** cite them in the text as "Fig. 1". Write "Figure 1" only at the start of a sentence. Captions go **below** figures. Number figures in the order they are first mentioned.
- **Tables:** the caption number is centered **above** the table as "TABLE I" (Roman numerals), with the descriptive caption text directly below it. Cite them in text as "Table I". Every table and figure must be mentioned in the text.
- **Equations:** number consecutively in parentheses flush right, (1), (2). The conference template asks for "(1)" in text, not "Eq. (1)", except at the start of a sentence ("Equation (1) is …"). Define every symbol right after the equation where it first appears.
- **Citations:** square brackets inside the sentence before punctuation: "…as shown in [3]." Several: "[3], [5]". Ranges: "[3]–[6]". Do not write "in reference [3]"; write "in [3]". The manual also discourages author names before the number ("in Smith [1]" should become "in [1]"). Many conference papers still write "Strack et al. [4]"; the draft does this for readability, so change it if your conference or reviewers follow the manual strictly.
- **Reference list:** numbered in the order of first citation (not alphabetical), formatted per the IEEE Reference Guide. Include DOIs where available. `paper/references.md` is already ordered this way.
- **Numbers and units:** SI units with a space before the unit ("900 s", "8 GB"). Per the manual, include the zero before a decimal point when it helps clarity, and do not add trailing zeros after the last significant digit. Advice: keep the same number of decimal places within one table.
- **Spelling and grammar:** IEEE follows Merriam-Webster for spelling (US English) and The Chicago Manual of Style for grammar questions the manual does not cover.

## 5. Figures and tables: practical rules

These are advice, not IEEE rules, except where marked:

- Export figures at **300 dpi or higher** PNG (our scripts already do), or as vector graphics (PDF/EMF/SVG) if the conference accepts them.
- Size every figure to fit **one column** or **the full page width** of the template. Check that axis labels are still readable at that size. Text in figures should be roughly the size of the caption text.
- Use colors that stay distinguishable in grayscale and for color-blind readers. Line styles and markers help when the paper is printed in black and white.
- Keep tables narrow enough for one column. For wide tables (e.g., all metrics × all models), use a table that spans both columns at the top or bottom of a page.
- Show uncertainty: mean ± std over seeds, or 95% CIs (the results CSVs contain both).

## 6. Ethics, data and AI-use disclosure

- **AI-generated content (IEEE policy, mandatory).** IEEE requires that the use of AI-generated content in an article (text, figures, images, code) be disclosed in the **Acknowledgment** section. The disclosure must name the AI system, identify the sections that contain AI-generated content, and briefly explain how much the AI system was used. Using AI only for editing and grammar falls outside the policy's intent, but disclosure is still recommended. This project used an AI assistant for code and drafting, so the paper **must** include a disclosure. Example (edit to match what actually happened):
  "The authors used Claude (Anthropic, model Opus 5.5) to assist with writing the analysis code and drafting Sections I–V. All AI-generated text and code were reviewed, verified and edited by the authors, who take full responsibility for the content."
- **Dataset license.** The data are CC BY 4.0. Cite the dataset [5] and the original study [4].
- **Ethics statement.** State that the study is retrospective and uses public, de-identified data. Some conferences require an explicit statement on ethics approval or exemption; check the call for papers.
- **Reporting guideline.** Check the paper against the TRIPOD+AI checklist [28]. Reviewers in clinical machine learning increasingly expect it, and it helps catch missing details (cohort, missing data, validation, calibration, limitations).
- **Do not overclaim.** Report modest and negative results as they are (see the honesty rules in `CLAUDE.md`).

## 7. Final PDF and submission

- **IEEE PDF eXpress.** Many IEEE conferences require the final PDF to be created or checked with IEEE PDF eXpress (https://ieee-pdf-express.org/). You need the **Conference ID** from your conference. Not every conference uses it; the conference invitation email tells you.
- **IEEE Xplore PDF requirements** (from the IEEE Author Center):
  - PDF version 1.4 or later (but not Acrobat 9, X or XI compatibility).
  - All fonts embedded or subset.
  - No password or security settings, no bookmarks or links, no attachments.
  - No crop marks, date stamps or other marks that are not in the official paper.
- **Copyright.** After acceptance, IEEE uses an electronic copyright form (eCF). The conference instructions say which copyright notice must appear at the bottom of the first page.
- **Blind review.** If the conference uses double-blind review, remove author names, affiliations, the GitHub URL and the Kaggle links from the submission version, and refer to your own work in the third person.

## 8. Pre-submission checklist

- [ ] Correct template (Letter or A4) and correct conference page limit
- [ ] All template guidance text deleted
- [ ] Title, author block and affiliation complete (or anonymized for blind review)
- [ ] Abstract 150–250 words, no citations or undefined abbreviations
- [ ] Index Terms present
- [ ] Every number in the text matches a file in `results/` (no TODO left)
- [ ] Every figure and table cited in the text, in numerical order, with captions in the right place
- [ ] Equations numbered and every symbol defined
- [ ] References in order of first citation, IEEE format, DOIs checked, verification notes removed
- [ ] Limitations stated (1999-2008 data, single source, no external validation, not for clinical use)
- [ ] AI-use disclosure in the Acknowledgment
- [ ] Data license and source cited; ethics statement if the conference requires one
- [ ] PDF passes IEEE PDF eXpress (if required), fonts embedded
- [ ] Repository URL and commit hash added (camera-ready only, if blind review)

## 9. Sources

- IEEE, "Manuscript Templates for Conference Proceedings": https://www.ieee.org/conferences/publishing/templates.html
- IEEE official templates on Overleaf: https://www.overleaf.com/gallery/tagged/ieee-official
- IEEE Author Center (Journals), "IEEE Editorial Style Manual for Authors": https://journals.ieeeauthorcenter.ieee.org/your-role-in-article-production/ieee-editorial-style-manual/
  (PDF used to check Section 4: https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Editorial-Style-Manual-for-Authors.pdf)
- IEEE Author Center (Journals), "Create the text of your article / IEEE Editorial Style Manual": https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/ieee-editorial-style-manual/ (includes the IEEE Reference Guide and the Mathematics Guide)
- IEEE Author Center (Conferences), "Meet IEEE Xplore Requirements": https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/meet-ieee-xplore-requirements/
- IEEE PDF eXpress: https://ieee-pdf-express.org/
- IEEE PDF Checker: https://www.ieee.org/publications/authors/pdf_checker.html
- IEEE Open, "Author Guidelines for Artificial Intelligence (AI)-Generated Text": https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/
- G. S. Collins et al., "TRIPOD+AI statement," BMJ, 2024: https://doi.org/10.1136/bmj-2023-078378 (checklist in the full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11025451/)

Items in Section 5 marked as advice (sizes, grayscale, CIs) are common practice in the field, not quoted IEEE rules. Section 3 lengths are a suggestion for this paper.
