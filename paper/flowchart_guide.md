# How to Draw Fig. 1 (Proposed Methodology Flowchart)

You have two options:

- **Option A (fastest):** open the ready-made, editable file `paper/figures/fig1_methodology.drawio` in draw.io and restyle it. All boxes, labels and arrows are already there.
- **Option B:** draw it from scratch in draw.io, Canva or Excalidraw using the exact layout and text below.

The finished figure must say the same thing as `paper/figures/fig1_methodology.png`, which is generated from the real numbers in `results/`. If you change a number, check it against `results/split.json` and `results/eda.json` first.

## 1. Open the tool

| Tool | How |
|---|---|
| draw.io (recommended) | Go to https://app.diagrams.net → *Open Existing Diagram* → choose `fig1_methodology.drawio`, or *Create New Diagram* → *Blank*. The desktop app works the same way. |
| Canva | *Create a design* → *Custom size* 1800 × 800 px. Use *Elements* → *Shapes* (rounded rectangle) and *Lines* (arrow). |
| Excalidraw | https://excalidraw.com. Set the font to *Normal* (not hand-drawn) in the style panel so the figure looks formal enough for IEEE. |

## 2. Page and grid

- Canvas about **1800 × 800 px** (wide). The figure spans both columns, which is about 18 cm wide in the IEEE template.
- Turn on the grid (draw.io: *View → Grid*, 10 px) so boxes align.
- Font: Arial or Helvetica, **11–12 pt** in draw.io. After shrinking to page width, the text should still be at least about 7 pt. If it isn't, use fewer words or larger boxes.

## 3. Colors (keep this meaning everywhere)

| Box type | Fill | Border | Used for |
|---|---|---|---|
| Data / preparation | `#EEEEEC` (light grey) | `#52514E` | dataset, cohort, split |
| Training only | `#DCE9F9` (light blue) | `#2A78D6` | training set, tuning, models, stacking, calibration |
| Test (used once) | `#FDE3D6` (light orange) | `#EB6834` | test set, evaluation |
| Explanation | `#DCF3EA` (light green) | `#1BAF7A` | SHAP |

Black text on all boxes; rounded corners; border width 1.5–2 px. Check the figure in grayscale (print preview or a screenshot set to grayscale): the box types should still be distinguishable by shade. If they aren't, add the words "train" and "test" to the boxes.

## 4. Layout: three rows, 12 boxes

Row 1, left to right (grey, then blue). All arrows point right.

1. **Diabetes 130-US Hospitals dataset** — 101,766 encounters, 71,518 patients
2. **Cohort and features** — exclude death/hospice (2,423); ICD-9 groups, 3-digit codes; medication counts; patient history
3. **Patient-level split** — stratified group 5-fold on patient ID (no shared patients)
4. **Training set** (blue) — 55,952 patients, 79,625 encounters

Row 2: all blue, inside a **dashed rectangle** labeled *"fitted on training patients only"*. The flow runs **right to left**, starting under box 4.

5. **Optuna tuning** — TPE sampler, PR-AUC on grouped folds (arrow down from box 4)
6. **Base learners** — XGBoost, LightGBM, CatBoost
7. **Patient-grouped 5-fold out-of-fold probabilities**
8. **LR meta-learner on logits (Eq. 1)** — or simple average
9. **Calibration** (none / Platt / isotonic) and **thresholds** (max-F1, recall 0.70)

Row 3, outside the dashed rectangle:

10. **Refitted models on test set** (orange) — ROC-AUC, PR-AUC, Brier, ECE, F1; patient-level bootstrap 95% CI. Arrow **down** from box 9.
11. **TreeSHAP per base model, weighted by meta-learner = exact stack attributions (Eq. 2)** (green). Arrow from box 10 to the right.
12. **Test set (used once)** (orange) — 14,038 patients, 19,718 encounters. Its arrow goes **into box 10**.

Two routed (orthogonal) arrows in **orange**, so they read as the test path:

- From box 3 (split) to box 12 (test set): go down a little, run right along the top edge of the dashed rectangle, then down the right side outside it into box 12. The test data must never pass *through* the training region.
- From box 12 to box 10: run along the bottom, under box 11, then up into box 10.

All other arrows: dark grey `#52514E`, 1.5 px, solid, filled arrowhead.

## 5. Step by step in draw.io

1. Drag a *Rounded Rectangle* from the General shapes. Right panel → *Style*: set Fill and Line colors from Section 3. *Text*: font 11 pt, centered.
2. Type the text; use Shift+Enter for line breaks. Size the box about 170–260 × 90 px.
3. Duplicate with Ctrl+D for the other boxes of the same type, then edit the text.
4. Align: select a row → *Arrange → Align → Middle*; then *Arrange → Distribute → Horizontally*.
5. Arrows: hover over a box, drag from the small blue arrow to the next box. Then *Style → Waypoints → Orthogonal* for the routed orange arrows.
6. Dashed region: draw a rectangle around row 2, set *Fill: none*, *Line: dashed*, color `#2A78D6`. Right-click → *To Back*. Add a small text label "fitted on training patients only" in italic blue.
7. Group everything (Ctrl+A, Ctrl+Shift+U) so it moves as one piece.

## 6. Export for the paper

- **draw.io:** *File → Export as → PNG*, *Zoom 300%* (or DPI 300), *Border width 10*, *Transparent background* off. Optionally also *Export as → PDF* or *SVG* for a sharp vector version, if the conference accepts it.
- **Canva:** *Share → Download → PNG*, size ×3, or *PDF Print*.
- **Excalidraw:** *Export image → PNG*, *Scale 3×*, background on.
- Save the file as `paper/figures/fig1_methodology.png`, replacing the generated one, and keep your source file (`.drawio`, Canva link, or `.excalidraw`) in `paper/figures/`.

## 7. Insert in Word

1. In the IEEE template, put the cursor where the draft says "[Insert Fig. 1 here, double-column width: …]".
2. A double-column figure goes at the top or bottom of a page. In Word, insert a one-column section break before and after it, or put it in a text box anchored to the top of the page.
3. *Insert → Pictures*, then right-click → *Size and Position* → width 18 cm (lock aspect ratio).
4. Caption **below** the figure, in the template's caption style: "Fig. 1. Proposed pipeline. Steps inside the dashed region are fitted on training patients only; the test set is used once for evaluation."
5. Delete the "[Insert …]" line.

## 8. Final checklist

- [ ] Every number matches `results/split.json` / `results/eda.json`.
- [ ] The test path (orange arrows) never crosses the dashed training region.
- [ ] Text is readable at 18 cm width; no text overflows a box.
- [ ] Equation references (Eq. 1, Eq. 2) match the numbering in Section IV.
- [ ] Readable in grayscale.
- [ ] Fig. 1 is cited in the text (Section IV, first sentence) before it appears.

The other figures (2–4) are already final in `paper/figures/`. Insert them the same way: Fig. 2 at double-column width, Figs. 3 and 4 at single-column width (8.9 cm).
