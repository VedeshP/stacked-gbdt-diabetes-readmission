"""Collect every number quoted in the paper from results/ into one table.

Usage (repo root):  python tools/paper_numbers.py
Output:             results/paper_numbers.json  (and printed), so each number in
                    paper/draft.md can be traced to the file it came from.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(os.environ.get("RESULTS_DIR_OVERRIDE", Path(__file__).resolve().parents[1] / "results"))
LABEL = {"number_inpatient": "prior-year inpatient visits",
         "discharge_disposition_id": "discharge disposition",
         "total_prior_visits": "total prior-year visits",
         "number_diagnoses": "number of diagnoses",
         "n_prior_enc": "number of earlier encounters in the data",
         "n_prior_readmit30": "number of earlier 30-day readmissions",
         "prev_readmitted": "outcome of the previous encounter",
         "diag_1_3d": "primary diagnosis (3-digit)",
         "number_emergency": "prior-year emergency visits",
         "payer_code": "payer code", "medical_specialty": "admitting specialty"}


def r3(x):
    return round(float(x), 3)


def pm(mean, std, nd=3):
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def write_tables(eda, split, comp, ci, base, fa, cal, imb):
    """Paper-ready tables (results/table*.csv), to paste into Word."""
    exc = eda["expired_or_hospice"]
    rows = [
        ("Full dataset", eda["n_encounters"], eda["n_patients"], eda["target_binary"]["n_positive"],
         eda["target_binary"]["positive_rate"]),
        ("After excluding death/hospice", exc["n_encounters_after_exclusion"], "",
         round(exc["n_encounters_after_exclusion"] * exc["positive_rate_after_exclusion"]),
         exc["positive_rate_after_exclusion"]),
        ("Training set", split["train"]["n_encounters"], split["train"]["n_patients"],
         split["train"]["n_positive"], split["train"]["positive_rate"]),
        ("Test set", split["test"]["n_encounters"], split["test"]["n_patients"],
         split["test"]["n_positive"], split["test"]["positive_rate"]),
    ]
    t1 = pd.DataFrame(rows, columns=["Subset", "Encounters", "Patients", "Readmitted <30 days",
                                     "Positive rate (%)"])
    t1["Positive rate (%)"] = (100 * t1["Positive rate (%)"]).round(2)
    t1.to_csv(R / "table1_cohort.csv", index=False)

    names = {"lr": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost",
             "lgbm": "LightGBM", "cat": "CatBoost", "average": "Simple average", "stack": "Stack (LR meta)"}
    rows = []
    for m in ["lr", "rf"]:
        b = base.loc[m]
        rows.append((names[m], f"{b['roc_auc']:.3f}", f"{b['pr_auc']:.3f}", f"{b['brier']:.4f}",
                     f"{b['ece']:.4f}", f"{b['f1']:.3f}",
                     f"{b['roc_auc_ci_low']:.3f}-{b['roc_auc_ci_high']:.3f}"))
    for m in ["xgb", "lgbm", "cat", "average", "stack"]:
        c = comp.loc[m]
        rows.append((names[m], pm(c["roc_auc_mean"], c["roc_auc_std"]), pm(c["pr_auc_mean"], c["pr_auc_std"]),
                     pm(c["brier_mean"], c["brier_std"], 4), pm(c["ece_mean"], c["ece_std"], 4),
                     pm(c["f1_mean"], c["f1_std"]),
                     f"{ci.loc[m, 'roc_auc_ci_low']:.3f}-{ci.loc[m, 'roc_auc_ci_high']:.3f}"))
    pd.DataFrame(rows, columns=["Model", "ROC-AUC", "PR-AUC", "Brier", "ECE", "F1 (max-F1 threshold)",
                                "ROC-AUC 95% CI"]).to_csv(R / "table2_main_results.csv", index=False)

    rows = []
    for name in fa.index:
        rows.append(("Features (LightGBM)", name, f"{fa.loc[name, 'test_roc_auc']:.3f}",
                     f"{fa.loc[name, 'test_pr_auc']:.3f}", f"{fa.loc[name, 'test_brier']:.4f}", ""))
    for _, r in imb.iterrows():
        rows.append(("Class weights", f"{names[r['model']]}, {'weighted' if r['class_weight'] else 'none'}",
                     f"{r['test_roc_auc']:.3f}", f"{r['test_pr_auc']:.3f}", f"{r['test_brier']:.4f}",
                     f"{r['test_ece']:.4f}"))
    for _, r in cal[cal["model"] == "stack"].iterrows():
        rows.append(("Calibration (stack)", r["calibration"], f"{r['roc_auc_mean']:.3f}",
                     f"{r['pr_auc_mean']:.3f}", f"{r['brier_mean']:.4f}", f"{r['ece_mean']:.4f}"))
    pd.DataFrame(rows, columns=["Ablation", "Setting", "ROC-AUC", "PR-AUC", "Brier", "ECE"]).to_csv(
        R / "table3_ablations.csv", index=False)


def main():
    out = {}
    eda = json.load(open(R / "eda.json"))
    split = json.load(open(R / "split.json"))
    out["n_features"] = split["n_features"]
    out["n_numeric"] = split["n_numeric_features"]
    out["n_categorical"] = split["n_categorical_features"]

    order = json.load(open(R / "encounter_order_check.json"))
    out["prev_label_no_share_pct"] = round(100 * order["label_of_previous_encounter_share"]["NO"], 1)
    out["n_with_earlier_encounter"] = order["n_encounters_with_earlier_encounter"]

    comp = pd.read_csv(R / "stacking_comparison.csv").set_index("model")
    for m in comp.index:
        for k in ["roc_auc", "pr_auc", "brier", "ece", "f1", "precision", "recall"]:
            out[f"{m}_{k}"] = r3(comp.loc[m, f"{k}_mean"])
            out[f"{m}_{k}_std"] = round(float(comp.loc[m, f"{k}_std"]), 4)
    singles = comp.loc[["xgb", "lgbm", "cat"]]
    out["best_single_by_roc"] = singles["roc_auc_mean"].idxmax()
    out["best_single_by_pr"] = singles["pr_auc_mean"].idxmax()

    ci = pd.read_csv(R / "stacking_test_ci_seed42.csv").set_index("model")
    for m in ci.index:
        out[f"{m}_roc_ci"] = f"{ci.loc[m, 'roc_auc_ci_low']:.3f}-{ci.loc[m, 'roc_auc_ci_high']:.3f}"
        out[f"{m}_pr_ci"] = f"{ci.loc[m, 'pr_auc_ci_low']:.3f}-{ci.loc[m, 'pr_auc_ci_high']:.3f}"

    base = pd.read_csv(R / "baselines_test.csv").set_index("model")
    for m in ["lr", "rf"]:
        for k in ["roc_auc", "pr_auc", "brier"]:
            out[f"{m}_{k}"] = r3(base.loc[m, k])
        out[f"{m}_roc_ci"] = f"{base.loc[m, 'roc_auc_ci_low']:.3f}-{base.loc[m, 'roc_auc_ci_high']:.3f}"
    cvb = pd.read_csv(R / "baselines_cv.csv")
    cvb = cvb[~cvb["class_weight"]].set_index("model")
    for m in cvb.index:
        out[f"{m}_cv_roc"] = r3(cvb.loc[m, "roc_auc_mean"])
        out[f"{m}_cv_roc_std"] = round(float(cvb.loc[m, "roc_auc_std"]), 4)

    corr = pd.read_csv(R / "base_oof_correlation_seed42.csv", index_col=0).to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]
    out["oof_corr_min"], out["oof_corr_max"] = round(off.min(), 2), round(off.max(), 2)
    w = pd.read_csv(R / "meta_weights_seed42.csv").set_index("base_model")["coef_on_logit"]
    out.update({f"w_{k}": round(float(v), 2) for k, v in w.items()})

    cal = pd.read_csv(R / "calibration_comparison.csv")
    for _, row in cal.iterrows():
        for k in ["brier", "ece", "pr_auc", "roc_auc"]:
            out[f"cal_{row['model']}_{row['calibration']}_{k}"] = round(float(row[f"{k}_mean"]), 4)

    imb = pd.read_csv(R / "imbalance_ablation.csv")
    wtd = imb[imb["class_weight"]]
    unw = imb[~imb["class_weight"]].set_index("model")
    out["weighted_brier_min"], out["weighted_brier_max"] = r3(wtd["test_brier"].min()), r3(wtd["test_brier"].max())
    out["unweighted_brier_min"], out["unweighted_brier_max"] = r3(unw["test_brier"].min()), r3(unw["test_brier"].max())
    d_roc = (wtd.set_index("model")["test_roc_auc"] - unw["test_roc_auc"])
    out["weighting_roc_change_min"], out["weighting_roc_change_max"] = round(d_roc.min(), 3), round(d_roc.max(), 3)

    op = pd.read_csv(R / "operating_points.csv").set_index("model")
    for m in op.index:
        out[f"op_{m}_recall"] = r3(op.loc[m, "recall_mean"])
        out[f"op_{m}_precision"] = r3(op.loc[m, "precision_mean"])
        out[f"op_{m}_flagged_pct"] = round(100 * op.loc[m, "flagged_fraction_mean"], 1)

    fa = pd.read_csv(R / "feature_ablation.csv").set_index("feature_set")
    for k, name in [("base", "base"), ("icd3", "base + ICD-9 3-digit"),
                    ("hist", "base + patient history"), ("full", "full")]:
        if name in fa.index:
            out[f"fa_{k}_roc"] = r3(fa.loc[name, "test_roc_auc"])
            out[f"fa_{k}_pr"] = r3(fa.loc[name, "test_pr_auc"])
            out[f"fa_{k}_roc_ci"] = f"{fa.loc[name, 'test_roc_auc_ci_low']:.3f}-{fa.loc[name, 'test_roc_auc_ci_high']:.3f}"
            out[f"fa_{k}_n"] = int(fa.loc[name, "n_features"])

    shap_stack = pd.read_csv(R / "shap_importance_stack.csv")
    out["shap_top5"] = [LABEL.get(f, f) for f in shap_stack["feature"].head(5)]
    add = json.load(open(R / "shap_additivity_check.json"))
    out["shap_additivity_max"] = f"{max(v for k, v in add.items() if k != 'n_test_encounters_checked'):.1e}"
    out["shap_n_checked"] = add["n_test_encounters_checked"]

    bp = json.load(open(R / "best_params.json"))
    out["trials_completed"] = {m: t["trials"].get("COMPLETE", 0) for m, t in bp["tuning"].items()}

    env = json.load(open(R / "env_kaggle.json")) if (R / "env_kaggle.json").exists() else {}
    out["cpu_count"] = env.get("cpu_count")
    out["git_commit"] = env.get("git_commit")
    rt = R / "runtime_kaggle.json"
    if rt.exists():
        secs = json.load(open(rt))
        out["runtime_min_total"] = round(sum(secs.values()) / 60)
        out["runtime_min_by_script"] = {k: round(v / 60, 1) for k, v in secs.items()}

    write_tables(eda, split, comp, ci, base, fa, cal, imb)

    with open(R / "paper_numbers.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
