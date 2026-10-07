"""03 - Tuned single-model baselines + class-weight ablation.

For each of LR, RF, XGBoost, LightGBM, CatBoost (each an sklearn Pipeline):
  1. Optuna TPE search on TRAIN only, objective = mean PR-AUC over the first
     TUNE_FOLDS patient-grouped folds (budget: N_TRIALS trials or TUNE_TIMEOUT s).
  2. Out-of-fold (OOF) predictions with 5-fold StratifiedGroupKFold on TRAIN.
     Decision threshold = max-F1 on OOF (never on test).
  3. Re-fit on all of TRAIN, evaluate once on TEST with patient-level bootstrap CIs.
  4. Ablation: repeat 2-3 with class weighting (scale_pos_weight / balanced).

Outputs:
  results/best_params.json                tuned hyperparameters + tuning metadata
  results/tuning_trials_<model>.csv       every Optuna trial
  results/baselines_cv.csv                OOF metrics, mean +- std over 5 folds
  results/baselines_test.csv              TEST metrics + 95% bootstrap CI (no class weights)
  results/imbalance_ablation.csv          no weighting vs class weights (CV + TEST)
  data/processed/preds_baselines.joblib   OOF and TEST probabilities
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from common import (N_BOOT, N_TRIALS, PROCESSED_DIR, SEED, TUNE_FOLDS, TUNE_TIMEOUT,
                    load_dataset, results_path, save_json, set_seed, train_test)
from evaluation import (METRICS, best_f1_threshold, classification_metrics, fold_ids_from,
                        fold_metrics, metrics_with_ci)
from models import MODEL_LABELS, MODEL_NAMES, grouped_cv, make_pipeline, tune


def main():
    set_seed()
    ds = load_dataset()
    num_cols, cat_cols = ds["num_cols"], ds["cat_cols"]
    X_tr, y_tr, g_tr, X_te, y_te, g_te = train_test(ds)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    cv = grouped_cv(SEED)
    fold_ids = fold_ids_from(cv, X_tr, y_tr, g_tr)

    # ---------------------------------------------------------------- tuning
    best = {"params": {}, "tuning": {}, "objective": f"mean PR-AUC over {TUNE_FOLDS} grouped folds",
            "n_trials_budget": N_TRIALS, "timeout_s_per_model": TUNE_TIMEOUT,
            "pos_weight": pos_weight}
    for name in MODEL_NAMES:
        t0 = time.time()
        study = tune(name, X_tr, y_tr, g_tr, num_cols, cat_cols,
                     n_trials=N_TRIALS, timeout=TUNE_TIMEOUT, n_folds=TUNE_FOLDS, seed=SEED)
        study.trials_dataframe().to_csv(results_path(f"tuning_trials_{name}.csv"), index=False)
        states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
        best["params"][name] = study.best_params
        best["tuning"][name] = {"best_cv_pr_auc": study.best_value, "trials": states,
                                "seconds": round(time.time() - t0, 1)}
        save_json(best, "best_params.json")
        print(f"[tune] {name}: best PR-AUC={study.best_value:.4f} trials={states} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- evaluation
    cv_rows, test_rows, ablation_rows, preds = [], [], [], {}
    for name in MODEL_NAMES:
        for weighted in (False, True):
            t0 = time.time()
            pipe = make_pipeline(name, num_cols, cat_cols, best["params"][name],
                                 class_weight=weighted, pos_weight=pos_weight, seed=SEED)
            oof = cross_val_predict(pipe, X_tr, y_tr, groups=g_tr, cv=cv,
                                    method="predict_proba")[:, 1]
            thr = best_f1_threshold(y_tr, oof)
            per_fold = fold_metrics(y_tr, oof, fold_ids, thr)

            pipe.fit(X_tr, y_tr)
            p_te = pipe.predict_proba(X_te)[:, 1]
            key = f"{name}{'_weighted' if weighted else ''}"
            preds[key] = {"oof": oof, "test": p_te, "threshold": thr}

            tag = {"model": name, "label": MODEL_LABELS[name], "class_weight": weighted}
            cv_row = {**tag, **{f"{m}_mean": per_fold[m].mean() for m in METRICS},
                      **{f"{m}_std": per_fold[m].std() for m in METRICS}, "threshold": thr}
            cv_rows.append(cv_row)

            if weighted:
                test = classification_metrics(y_te, p_te, thr)
            else:
                test = metrics_with_ci(y_te, p_te, g_te, thr, N_BOOT, SEED)
                test_rows.append({**tag, **test})
            ablation_rows.append({**tag,
                                  **{f"cv_{m}_mean": cv_row[f"{m}_mean"] for m in METRICS},
                                  **{f"cv_{m}_std": cv_row[f"{m}_std"] for m in METRICS},
                                  **{f"test_{m}": test[m] for m in METRICS},
                                  "threshold": thr})
            print(f"[eval] {key}: OOF ROC-AUC={cv_row['roc_auc_mean']:.4f}"
                  f"+-{cv_row['roc_auc_std']:.4f} PR-AUC={cv_row['pr_auc_mean']:.4f} | "
                  f"TEST ROC-AUC={test['roc_auc']:.4f} PR-AUC={test['pr_auc']:.4f} "
                  f"Brier={test['brier']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(cv_rows).to_csv(results_path("baselines_cv.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(results_path("baselines_test.csv"), index=False)
    pd.DataFrame(ablation_rows).to_csv(results_path("imbalance_ablation.csv"), index=False)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"y_train": y_tr, "y_test": y_te, "fold_ids": fold_ids, "preds": preds},
                PROCESSED_DIR / "preds_baselines.joblib")
    print("done: baselines_cv.csv, baselines_test.csv, imbalance_ablation.csv")


if __name__ == "__main__":
    main()
