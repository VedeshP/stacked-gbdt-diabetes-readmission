"""05 - SHAP explanations for the base learners and the stack.

Tree SHAP values for XGBoost / LightGBM / CatBoost are in log-odds. The stack's
logit is  b + sum_i w_i * logit(p_i), so the stack's per-feature attribution is
exactly  sum_i w_i * phi_i  once each model's one-hot columns are summed back
to their source feature (SHAP is additive).

Outputs (TEST sample of N_SHAP rows):
  results/shap_importance_<model>.csv          mean |SHAP| per transformed feature
  results/shap_importance_grouped_<model>.csv  mean |SHAP| per original feature
  results/shap_importance_stack.csv            stack attribution per original feature
  results/shap_summary_<best>.png              beeswarm for the best single model
  results/shap_top_features_<best>.png         top-20 grouped bar for the best single model
  results/shap_top_features_stack.png          top-20 grouped bar for the stack
"""
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from common import (FAST, SEED, STACKER_FILE, load_dataset, results_path, set_seed,
                    setup_matplotlib, train_test)
from evaluation import plot_hbar
from features import original_feature
from models import MODEL_LABELS

N_SHAP = 300 if FAST else 2000
warnings.filterwarnings("ignore", message=".*LightGBM binary classifier with TreeExplainer.*")


def shap_values(name, pipe, X):
    """Return (shap matrix, transformed DataFrame) in log-odds units."""
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    Xt = pre.transform(X)
    names = pre.get_feature_names_out()
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = pd.DataFrame(np.asarray(Xt) if not isinstance(Xt, pd.DataFrame) else Xt,
                      columns=names, index=X.index)
    if name == "cat":
        from catboost import Pool
        sv = model.model_.get_feature_importance(Pool(Xt, cat_features=model.cat_features_),
                                                 type="ShapValues")
        return sv[:, :-1], Xt
    sv = shap.TreeExplainer(model).shap_values(Xt)
    if isinstance(sv, list):  # older LightGBM/shap return [neg, pos]
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xt


def grouped(sv, columns, input_cols):
    """Sum SHAP columns that come from the same original feature."""
    owner = [original_feature(c, input_cols) for c in columns]
    return pd.DataFrame(sv, columns=columns).T.groupby(owner, sort=False).sum().T


def main():
    set_seed()
    plt = setup_matplotlib()
    ds = load_dataset()
    _, _, _, X_te, _, _ = train_test(ds)
    input_cols = list(ds["X"].columns)
    stacker = joblib.load(STACKER_FILE)
    sample = X_te.sample(n=min(N_SHAP, len(X_te)), random_state=SEED)

    comp = pd.read_csv(results_path("stacking_comparison.csv"))
    singles = comp[comp["model"].isin(stacker.names_)]
    best = singles.sort_values("pr_auc_mean", ascending=False)["model"].iloc[0]

    weights = stacker.final_estimator_.coef_.ravel()
    stack_phi = None
    for name, pipe, w in zip(stacker.names_, stacker.estimators_, weights):
        sv, Xt = shap_values(name, pipe, sample)
        imp = pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(
            results_path(f"shap_importance_{name}.csv"), index=False)

        g = grouped(sv, Xt.columns, input_cols)
        gimp = g.abs().mean().sort_values(ascending=False)
        gimp.rename("mean_abs_shap").rename_axis("feature").reset_index().to_csv(
            results_path(f"shap_importance_grouped_{name}.csv"), index=False)

        contrib = w * g.reindex(columns=input_cols, fill_value=0.0)
        stack_phi = contrib if stack_phi is None else stack_phi + contrib

        if name == best:
            shap.summary_plot(sv, Xt, max_display=20, show=False, plot_size=(7, 6),
                              rng=np.random.default_rng(SEED))
            plt.gcf().savefig(results_path(f"shap_summary_{name}.png"))
            plt.close("all")
            plot_hbar(plt, gimp.index, gimp.values, f"shap_top_features_{name}.png",
                      f"Mean |SHAP| (log-odds), {MODEL_LABELS[name]}")
        print(f"[shap] {name}: top-5 = {list(gimp.index[:5])}", flush=True)

    stack_imp = stack_phi.abs().mean().sort_values(ascending=False)
    stack_imp.rename("mean_abs_shap").rename_axis("feature").reset_index().to_csv(
        results_path("shap_importance_stack.csv"), index=False)
    plot_hbar(plt, stack_imp.index, stack_imp.values, "shap_top_features_stack.png",
              "Mean |SHAP| of stacked model (log-odds)")
    print(f"[shap] stack: top-5 = {list(stack_imp.index[:5])}; best single = {best}")


if __name__ == "__main__":
    main()
