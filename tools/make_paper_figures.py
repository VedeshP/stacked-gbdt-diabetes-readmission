"""Collect the paper figures in paper/figures/ with final names (300 dpi PNG).

Fig. 1 is drawn by tools/make_methodology_figure.py. Fig. 4 is redrawn here with
readable feature names; Figs. 2-3 are copied from results/ unchanged.

Usage (repo root):  python tools/make_paper_figures.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import PALETTE, setup_matplotlib  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "paper" / "figures"

NAMES = {
    "discharge_disposition_id": "Discharge disposition", "number_inpatient": "Inpatient visits (prior year)",
    "payer_code": "Payer code", "prev_readmitted": "Outcome of previous encounter*",
    "diag_1_3d": "Primary diagnosis (3-digit)", "total_prior_visits": "Total visits (prior year)",
    "n_prior_enc": "Number of earlier encounters*", "diag_3_3d": "Third diagnosis (3-digit)",
    "diag_1_group": "Primary diagnosis group", "num_medications": "Number of medications",
    "diag_2_3d": "Secondary diagnosis (3-digit)", "medical_specialty": "Admitting specialty",
    "n_drugs_active": "Active diabetes drugs", "number_diagnoses": "Number of diagnoses",
    "num_lab_procedures": "Number of lab procedures", "age": "Age",
    "prev_time_in_hospital": "Previous length of stay*", "time_in_hospital": "Length of stay",
    "number_emergency": "Emergency visits (prior year)", "admission_source_id": "Admission source",
    "n_prior_readmit30": "Earlier 30-day readmissions*", "prev_discharge": "Previous discharge disposition*",
}


def fig4(path, top=15):
    plt = setup_matplotlib()
    imp = pd.read_csv(RES / "shap_importance_stack.csv").head(top)
    labels = [NAMES.get(f, f) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    y = range(len(imp))
    ax.barh(list(y), imp["mean_abs_shap"], height=0.62, color=PALETTE["series1"])
    ax.set_yticks(list(y), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| of the stack (log-odds)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.02, "* patient-history feature", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.5, color=PALETTE["text_muted"])
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RES / "fig1_methodology.png", OUT / "fig1_methodology.png")
    shutil.copy2(RES / "roc_pr_curves_seed42.png", OUT / "fig2_roc_pr_curves.png")
    shutil.copy2(RES / "reliability_seed42.png", OUT / "fig3_reliability.png")
    fig4(OUT / "fig4_shap_stack.png")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
