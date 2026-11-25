"""Draw Fig. 1 (proposed methodology) as a 300 dpi PNG and as an editable draw.io file.

Numbers are read from results/eda.json and results/split.json.

Usage (repo root):  python tools/make_methodology_figure.py
Outputs:            results/fig1_methodology.png
                    paper/figures/fig1_methodology.drawio   (open in https://app.diagrams.net)
"""
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import PALETTE, setup_matplotlib  # noqa: E402

eda = json.load(open(ROOT / "results" / "eda.json", encoding="utf-8"))
split = json.load(open(ROOT / "results" / "split.json", encoding="utf-8"))
tr, te = split["train"], split["test"]

FILL = {"data": "#eeeeec", "train": "#dce9f9", "test": "#fde3d6", "explain": "#dcf3ea"}
EDGE = {"data": PALETTE["text_muted"], "train": PALETTE["series1"],
        "test": PALETTE["series2"], "explain": PALETTE["series3"]}

# (id, kind, x, y, w, h, text) in a 100 x 40 canvas, y measured downward
BOXES = [
    ("data", "data", 1, 1, 17, 10,
     f"Diabetes 130-US\nHospitals dataset\n{eda['n_encounters']:,} encounters\n{eda['n_patients']:,} patients"),
    ("prep", "data", 21, 1, 26, 10,
     f"Cohort and features\nexclude death/hospice ({eda['expired_or_hospice']['n_encounters']:,})\nICD-9 groups, 3-digit codes,\nmedication counts,\npatient history"),
    ("split", "data", 51, 1, 26, 10,
     "Patient-level split\nstratified group 5-fold\non patient ID\n(no shared patients)"),
    ("trainset", "train", 81, 1, 18, 10,
     f"Training set\n{tr['n_patients']:,} patients\n{tr['n_encounters']:,} encounters"),
    ("tune", "train", 81, 16, 18, 9,
     "Optuna tuning\n(TPE sampler)\nPR-AUC on\ngrouped folds"),
    ("base", "train", 61, 16, 17, 9,
     "Base learners\nXGBoost\nLightGBM\nCatBoost"),
    ("oof", "train", 41, 16, 17, 9,
     "Patient-grouped\n5-fold out-of-fold\nprobabilities"),
    ("meta", "train", 21, 16, 17, 9,
     "LR meta-learner\non logits (Eq. 1)\nor simple average"),
    ("calib", "train", 1, 16, 17, 9,
     "Calibration\n(none/Platt/isotonic),\nthresholds (max-F1,\nrecall 0.70)"),
    ("testset", "test", 74, 30, 25, 9,
     f"Test set (used once)\n{te['n_patients']:,} patients\n{te['n_encounters']:,} encounters"),
    ("eval", "test", 1, 30, 34, 9,
     "Refitted models on test set\nROC-AUC, PR-AUC, Brier, ECE, F1\npatient-level bootstrap 95% CI"),
    ("shap", "explain", 39, 30, 31, 9,
     "TreeSHAP per base model\nweighted by meta-learner\n= exact stack attributions (Eq. 2)"),
]
EDGES = [("data", "prep"), ("prep", "split"), ("split", "trainset"), ("trainset", "tune"),
         ("tune", "base"), ("base", "oof"), ("oof", "meta"), ("meta", "calib"),
         ("calib", "eval"), ("eval", "shap")]
SPLIT_TO_TEST = ("split", "testset")
TEST_TO_EVAL = ("testset", "eval")


def anchor(a, b):
    """Start/end points on box borders between box a and box b."""
    (_, _, ax, ay, aw, ah, _), (_, _, bx, by, bw, bh, _) = a, b
    acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
    if abs(acy - bcy) < 1:            # same row: horizontal arrow
        return ((ax + aw, acy), (bx, bcy)) if bcx > acx else ((ax, acy), (bx + bw, bcy))
    lo, hi = max(ax, bx), min(ax + aw, bx + bw)
    if hi - lo > 2:                   # boxes overlap horizontally: vertical arrow in the overlap
        xm = (lo + hi) / 2
        return ((xm, ay + ah), (xm, by)) if bcy > acy else ((xm, ay), (xm, by + bh))
    return (acx, ay + ah), (bcx, by)


def render_png(path):
    plt = setup_matplotlib()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.16, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(43.5, 0)
    ax.axis("off")
    boxes = {b[0]: b for b in BOXES}
    # dashed region: everything fitted on training patients only
    ax.add_patch(FancyBboxPatch((0.3, 14.3), 99.4, 11.9, boxstyle="round,pad=0,rounding_size=1",
                                fill=False, ls="--", lw=0.8, ec=EDGE["train"]))
    ax.text(0.8, 14.0, "fitted on training patients only", ha="left", va="bottom", fontsize=5.6,
            color=EDGE["train"], style="italic")
    for bid, kind, x, y, w, h, text in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=FILL[kind], ec=EDGE[kind], lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.8,
                color=PALETTE["text"], linespacing=1.15)
    for a, b in EDGES:
        (x1, y1), (x2, y2) = anchor(boxes[a], boxes[b])
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=PALETTE["text_muted"]))
    # split -> test set: route along the right edge, outside the training region
    route = [(64, 11), (64, 12.6), (101.2, 12.6), (101.2, 34.5), (99, 34.5)]
    # test set -> evaluation: route below the bottom row
    route2 = [(86.5, 39), (86.5, 42), (18, 42), (18, 39)]
    for pts in (route, route2):
        ax.plot(*zip(*pts[:-1]), color=EDGE["test"], lw=0.8, clip_on=False)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=EDGE["test"], clip_on=False))
    fig.savefig(path)
    plt.close(fig)


def drawio_xml(path):
    scale = 9  # canvas units -> pixels
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for bid, kind, x, y, w, h, text in BOXES:
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={FILL[kind]};strokeColor={EDGE[kind]};"
                 "fontSize=11;fontFamily=Helvetica;")
        label = escape(text).replace("\n", "&lt;br&gt;")
        cells.append(f'<mxCell id="{bid}" value="{label}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x * scale}" y="{y * scale}" width="{w * scale}" '
                     f'height="{h * scale}" as="geometry"/></mxCell>')
    for i, (a, b) in enumerate(EDGES + [SPLIT_TO_TEST, TEST_TO_EVAL]):
        cells.append(f'<mxCell id="e{i}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
                     f'endArrow=block;endFill=1;" edge="1" parent="1" source="{a}" target="{b}">'
                     '<mxGeometry relative="1" as="geometry"/></mxCell>')
    xml = ('<mxfile host="app.diagrams.net"><diagram name="Fig. 1 methodology">'
           '<mxGraphModel dx="1000" dy="600" grid="1" gridSize="10" page="1" pageWidth="1000" '
           'pageHeight="400"><root>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    png = ROOT / "results" / "fig1_methodology.png"
    render_png(png)
    drawio_xml(ROOT / "paper" / "figures" / "fig1_methodology.drawio")
    print(f"wrote {png} and paper/figures/fig1_methodology.drawio")
