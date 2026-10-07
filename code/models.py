"""Model pipelines, Optuna search spaces, tuning, and the grouped OOF stacking estimator.

Every model is an sklearn Pipeline([("pre", ColumnTransformer), ("model", estimator)]),
so all fitted preprocessing is re-fit inside each CV fold (no leakage).
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from common import CV_FOLDS, FAST, N_JOBS, SEED, gpu_available
from features import native_preprocessor, onehot_preprocessor

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

MODEL_NAMES = ["lr", "rf", "xgb", "lgbm", "cat"]
BASE_LEARNERS = ["xgb", "lgbm", "cat"]
MODEL_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
                "lgbm": "LightGBM", "cat": "CatBoost", "average": "Simple average",
                "stack": "Stacked (LR meta)"}

N_EST_MAX = 60 if FAST else 1000
N_EST_MIN = 30 if FAST else 200
N_EST_STEP = 10 if FAST else 100

# Starting points; enqueued as Optuna trial 0 so tuning never does worse than these.
DEFAULT_PARAMS = {
    "lr": {"C": 1.0},
    "rf": {"n_estimators": N_EST_MIN, "max_depth": 12, "min_samples_leaf": 20, "max_features": 0.2},
    "xgb": {"n_estimators": N_EST_MIN * 2 if not FAST else N_EST_MAX, "learning_rate": 0.05,
            "max_depth": 5, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_lambda": 1.0, "reg_alpha": 0.01},
    "lgbm": {"n_estimators": N_EST_MIN * 2 if not FAST else N_EST_MAX, "learning_rate": 0.05,
             "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8,
             "colsample_bytree": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.01},
    "cat": {"iterations": N_EST_MIN * 3 if not FAST else N_EST_MAX, "learning_rate": 0.05,
            "depth": 6, "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 0.5},
}


def suggest_params(name, trial):
    n_est = lambda key: trial.suggest_int(key, N_EST_MIN, N_EST_MAX, step=N_EST_STEP)  # noqa: E731
    if name == "lr":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", N_EST_MIN, max(N_EST_MIN, N_EST_MAX // 2),
                                                  step=N_EST_STEP),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
                "max_features": trial.suggest_float("max_features", 0.05, 0.5)}
    if name == "xgb":
        return {"n_estimators": n_est("n_estimators"),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}
    if name == "lgbm":
        return {"n_estimators": n_est("n_estimators"),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}
    if name == "cat":
        return {"iterations": n_est("iterations"),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0)}
    raise ValueError(name)


def make_pipeline(name, num_cols, cat_cols, params=None, class_weight=False,
                  pos_weight=1.0, seed=SEED):
    """Build Pipeline(pre, model). class_weight=True re-weights the positive class."""
    params = dict(DEFAULT_PARAMS[name] if params is None else params)
    if name == "lr":
        pre = onehot_preprocessor(num_cols, cat_cols, scale=True)
        model = LogisticRegression(**params, max_iter=3000, random_state=seed,
                                   class_weight="balanced" if class_weight else None)
    elif name == "rf":
        pre = onehot_preprocessor(num_cols, cat_cols)
        model = RandomForestClassifier(**params, n_jobs=N_JOBS, random_state=seed,
                                       class_weight="balanced_subsample" if class_weight else None)
    elif name == "xgb":
        from xgboost import XGBClassifier
        pre = onehot_preprocessor(num_cols, cat_cols)
        model = XGBClassifier(**params, tree_method="hist",
                              device="cuda" if gpu_available() else "cpu",
                              eval_metric="logloss", n_jobs=N_JOBS, random_state=seed,
                              scale_pos_weight=pos_weight if class_weight else 1.0)
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        pre = onehot_preprocessor(num_cols, cat_cols)
        model = LGBMClassifier(**params, subsample_freq=1, n_jobs=N_JOBS, random_state=seed,
                               verbose=-1, scale_pos_weight=pos_weight if class_weight else 1.0)
    elif name == "cat":
        pre = native_preprocessor(num_cols, cat_cols)
        model = CatBoostNative(params=params, class_weight=class_weight, seed=seed)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("model", model)])


class CatBoostNative(ClassifierMixin, BaseEstimator):
    """sklearn-clonable CatBoost with native categoricals.

    CatBoostClassifier rewrites its `cat_features` argument, which breaks
    sklearn.clone(); here categorical columns (object dtype) are detected at fit.
    """

    def __init__(self, params=None, class_weight=False, seed=SEED):
        self.params = params
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.cat_features_ = [c for c in X.columns if X[c].dtype == object]
        self.model_ = CatBoostClassifier(
            **(self.params or {}), cat_features=self.cat_features_, random_seed=self.seed,
            thread_count=N_JOBS, verbose=0, allow_writing_files=False,
            auto_class_weights="Balanced" if self.class_weight else None)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def grouped_cv(seed=SEED, n_splits=CV_FOLDS):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def tune(name, X, y, groups, num_cols, cat_cols, n_trials, timeout, n_folds, seed=SEED):
    """Optuna TPE search maximising mean PR-AUC over the first n_folds grouped folds."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits = list(grouped_cv(seed).split(X, y, groups))[:n_folds]

    def objective(trial):
        params = suggest_params(name, trial)
        scores = []
        for i, (tr, va) in enumerate(splits):
            pipe = make_pipeline(name, num_cols, cat_cols, params, seed=seed)
            pipe.fit(X.iloc[tr], y[tr])
            scores.append(average_precision_score(y[va], pipe.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=f"tune_{name}",
    )
    study.enqueue_trial(DEFAULT_PARAMS[name])
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class GroupOOFStackingClassifier(ClassifierMixin, BaseEstimator):
    """Stacking with out-of-fold base predictions from a *grouped* splitter.

    sklearn's StackingClassifier cannot pass `groups` to its internal CV, so the
    same patient could land in both a base model's training fold and the fold it
    predicts. Here cross_val_predict receives groups explicitly. The meta-learner
    is fit on the logits of the OOF probabilities; base models are then re-fit on
    all training data for inference.
    """

    def __init__(self, estimators, final_estimator=None, cv=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y, groups=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        cv = self.cv if self.cv is not None else grouped_cv()
        self.oof_ = np.column_stack([
            cross_val_predict(clone(est), X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
            for _, est in self.estimators
        ])
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        meta = self.final_estimator if self.final_estimator is not None else LogisticRegression()
        self.final_estimator_ = clone(meta).fit(logit(self.oof_), y)
        self.names_ = [n for n, _ in self.estimators]
        return self

    def base_predict_proba(self, X):
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(logit(self.base_predict_proba(X)))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
