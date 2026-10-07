"""04 - Out-of-fold stacking, averaging ablation, calibration, and thresholds.

For each seed in SEEDS (CV fold assignment + model seeds change; TEST is fixed):
  1. GroupOOFStackingClassifier: XGBoost, LightGBM, CatBoost (tuned params from
     03) produce 5-fold patient-grouped OOF probabilities on TRAIN; a logistic
     regression meta-learner is fit on their logits; bases are re-fit on TRAIN.
  2. Candidates: each single base model, simple average, stack.
     Stack OOF probabilities are cross-fitted (meta-learner itself run through
     the same grouped CV) so thresholds/calibrators never see their own labels.
  3. Calibration: none / sigmoid (Platt) / isotonic, fit on OOF only.
  4. Thresholds from OOF: max-F1, and recall >= RECALL_TARGET.
  5. TEST metrics for every candidate x calibration.

Outputs:
  results/stacking_by_seed.csv           every seed x candidate x calibration
  results/stacking_comparison.csv        uncalibrated, mean +- std over seeds (ablation table)
  results/calibration_comparison.csv     Brier / ECE / AUC per calibration, mean +- std
  results/operating_points.csv           metrics at recall-target threshold, mean +- std
  results/stacking_test_ci_seed42.csv    TEST metrics + 95% patient bootstrap CI, seed 42
  results/meta_weights_seed42.csv        meta-learner coefficients
  results/base_oof_correlation_seed42.csv
  results/roc_pr_curves_seed42.png
  results/reliability_seed42.png
  data/processed/stacker_seed42.joblib   fitted stacker (used by 05_explain.py)
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from common import (N_BOOT, SEED, SEEDS, STACKER_FILE, load_dataset, load_json, results_path,
                    set_seed, setup_matplotlib, train_test)
from evaluation import (METRICS, RECALL_TARGET, best_f1_threshold, classification_metrics,
                        fit_calibrator, mean_std, metrics_with_ci, plot_reliability, plot_roc_pr,
                        recall_threshold)
from models import (BASE_LEARNERS, MODEL_LABELS, GroupOOFStackingClassifier, grouped_cv, logit,
                    make_pipeline)

CALIBRATIONS = ["none", "sigmoid", "isotonic"]


def meta_learner():
    return LogisticRegression(C=1.0, max_iter=1000)


def main():
    set_seed()
    plt = setup_matplotlib()
    ds = load_dataset()
    num_cols, cat_cols = ds["num_cols"], ds["cat_cols"]
    X_tr, y_tr, g_tr, X_te, y_te, g_te = train_test(ds)
    params = load_json("best_params.json")["params"]

    rows, op_rows = [], []
    for seed in SEEDS:
        t0 = time.time()
        set_seed(seed)
        cv = grouped_cv(seed)
        bases = [(n, make_pipeline(n, num_cols, cat_cols, params[n], seed=seed))
                 for n in BASE_LEARNERS]
        stacker = GroupOOFStackingClassifier(bases, meta_learner(), cv).fit(X_tr, y_tr, groups=g_tr)

        oof_base = stacker.oof_
        te_base = stacker.base_predict_proba(X_te)
        oof_stack = cross_val_predict(meta_learner(), logit(oof_base), y_tr, groups=g_tr, cv=cv,
                                      method="predict_proba")[:, 1]
        te_stack = stacker.predict_proba(X_te)[:, 1]

        candidates = {n: (oof_base[:, i], te_base[:, i]) for i, n in enumerate(BASE_LEARNERS)}
        candidates["average"] = (oof_base.mean(axis=1), te_base.mean(axis=1))
        candidates["stack"] = (oof_stack, te_stack)

        calibrated_test = {}
        for cand, (p_oof, p_te) in candidates.items():
            for calib in CALIBRATIONS:
                f = fit_calibrator(calib, p_oof, y_tr)
                q_oof, q_te = f(p_oof), f(p_te)
                calibrated_test[(cand, calib)] = q_te
                tag = {"seed": seed, "model": cand, "label": MODEL_LABELS[cand],
                       "calibration": calib}
                rows.append({**tag, **classification_metrics(y_te, q_te,
                                                             best_f1_threshold(y_tr, q_oof))})
                if calib == "none":
                    op_rows.append({**tag, "recall_target": RECALL_TARGET,
                                    **classification_metrics(y_te, q_te,
                                                             recall_threshold(y_tr, q_oof))})

        if seed == SEED:
            joblib.dump(stacker, STACKER_FILE)
            coef = stacker.final_estimator_.coef_.ravel()
            pd.DataFrame({"base_model": BASE_LEARNERS, "coef_on_logit": coef,
                          "intercept": stacker.final_estimator_.intercept_[0]}
                         ).to_csv(results_path("meta_weights_seed42.csv"), index=False)
            pd.DataFrame(np.corrcoef(logit(oof_base).T), index=BASE_LEARNERS,
                         columns=BASE_LEARNERS).to_csv(results_path("base_oof_correlation_seed42.csv"))

            ci_rows = []
            for cand, (p_oof, p_te) in candidates.items():
                thr = best_f1_threshold(y_tr, p_oof)
                ci_rows.append({"model": cand, "label": MODEL_LABELS[cand],
                                **metrics_with_ci(y_te, p_te, g_te, thr, N_BOOT, SEED)})
            pd.DataFrame(ci_rows).to_csv(results_path("stacking_test_ci_seed42.csv"), index=False)

            plot_roc_pr(plt, y_te, {MODEL_LABELS[c]: p for c, (_, p) in candidates.items()},
                        "roc_pr_curves_seed42.png", " (test)")
            best_single = max(BASE_LEARNERS, key=lambda n: np.mean(
                [r["pr_auc"] for r in rows if r["model"] == n and r["calibration"] == "none"]))
            plot_reliability(plt, y_te, {
                f"{MODEL_LABELS[best_single]} (raw)": calibrated_test[(best_single, "none")],
                "Stack (raw)": calibrated_test[("stack", "none")],
                "Stack + Platt": calibrated_test[("stack", "sigmoid")],
                "Stack + isotonic": calibrated_test[("stack", "isotonic")],
            }, "reliability_seed42.png")

        last = {r["model"]: r for r in rows if r["seed"] == seed and r["calibration"] == "none"}
        print(f"[seed {seed}] " + " | ".join(
            f"{m}: AUC={last[m]['roc_auc']:.4f} PR={last[m]['pr_auc']:.4f}" for m in last)
            + f" ({time.time() - t0:.0f}s)", flush=True)

    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(results_path("stacking_by_seed.csv"), index=False)

    raw = by_seed[by_seed["calibration"] == "none"]
    comp = mean_std(raw, ["model", "label"])
    comp.insert(2, "n_seeds", len(SEEDS))
    comp.to_csv(results_path("stacking_comparison.csv"), index=False)

    mean_std(by_seed, ["model", "label", "calibration"], ["brier", "ece", "roc_auc", "pr_auc"]
             ).to_csv(results_path("calibration_comparison.csv"), index=False)
    mean_std(pd.DataFrame(op_rows), ["model", "label"], METRICS + ["flagged_fraction"]
             ).to_csv(results_path("operating_points.csv"), index=False)

    print(comp[["label", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
                "brier_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
