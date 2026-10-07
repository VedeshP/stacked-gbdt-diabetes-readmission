"""Metrics, threshold selection, calibration, patient-level bootstrap, and plots."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from common import PALETTE, SEED, results_path

METRICS = ["roc_auc", "pr_auc", "brier", "ece", "precision", "recall", "f1"]
RECALL_TARGET = 0.70


# ------------------------------------------------------------------ thresholds

def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def recall_threshold(y, p, target=RECALL_TARGET):
    """Highest threshold whose recall is still >= target."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok.max()]) if len(ok) else float(thr.min())


# ------------------------------------------------------------------ metrics

def ece(y, p, n_bins=10, w=None):
    """Expected calibration error with equal-frequency bins."""
    w = np.ones_like(p) if w is None else w
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    total = w.sum()
    return float(sum(abs(np.average(y[b], weights=w[b]) - np.average(p[b], weights=w[b]))
                     * w[b].sum() / total for b in bins if w[b].sum() > 0))


def classification_metrics(y, p, threshold, w=None):
    y = np.asarray(y)
    p = np.asarray(p)
    w = np.ones_like(p, dtype=float) if w is None else w
    pred = p >= threshold
    tp = np.sum(w * (pred & (y == 1)))
    fp = np.sum(w * (pred & (y == 0)))
    fn = np.sum(w * (~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "roc_auc": roc_auc_score(y, p, sample_weight=w),
        "pr_auc": average_precision_score(y, p, sample_weight=w),
        "brier": brier_score_loss(y, p, sample_weight=w),
        "ece": ece(y, p, w=w),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "flagged_fraction": float(np.average(pred, weights=w)),
    }


def bootstrap_ci(y, p, groups, threshold, n_boot, seed=SEED, alpha=0.05):
    """Patient-level (cluster) bootstrap: resample patients, weight rows by draw count."""
    y, p = np.asarray(y), np.asarray(p)
    codes, uniq = pd.factorize(np.asarray(groups))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        w = counts[codes].astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        draws.append(classification_metrics(y, p, threshold, w=w))
    df = pd.DataFrame(draws)
    return {m: (float(df[m].quantile(alpha / 2)), float(df[m].quantile(1 - alpha / 2)))
            for m in METRICS}


def metrics_with_ci(y, p, groups, threshold, n_boot, seed=SEED):
    row = classification_metrics(y, p, threshold)
    for m, (lo, hi) in bootstrap_ci(y, p, groups, threshold, n_boot, seed).items():
        row[f"{m}_ci_low"], row[f"{m}_ci_high"] = lo, hi
    return row


def fold_metrics(y, p, fold_ids, threshold):
    """Metrics per CV fold -> DataFrame (one row per fold)."""
    rows = []
    for f in np.unique(fold_ids):
        m = fold_ids == f
        rows.append({"fold": int(f), **classification_metrics(y[m], p[m], threshold)})
    return pd.DataFrame(rows)


def fold_ids_from(cv, X, y, groups):
    ids = np.empty(len(y), dtype=int)
    for k, (_, va) in enumerate(cv.split(X, y, groups)):
        ids[va] = k
    return ids


def mean_std(df, by, metrics=METRICS):
    agg = df.groupby(by, sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


# ------------------------------------------------------------------ calibration

def fit_calibrator(kind, p, y):
    """Return f(p) -> calibrated p. kind in {'none', 'sigmoid', 'isotonic'}."""
    if kind == "none":
        return lambda q: np.asarray(q)
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return iso.predict
    if kind == "sigmoid":
        z = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))  # noqa: E731
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z(p).reshape(-1, 1), y)
        return lambda q: lr.predict_proba(z(np.asarray(q)).reshape(-1, 1))[:, 1]
    raise ValueError(kind)


# ------------------------------------------------------------------ plots

SERIES_COLORS = [PALETTE[f"series{i}"] for i in range(1, 6)]


def plot_roc_pr(plt, y, preds: dict, fname, title_suffix=""):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for (label, p), color in zip(preds.items(), SERIES_COLORS):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} ({roc_auc_score(y, p):.3f})")
        a2.plot(rec, prec, color=color, lw=1.5, label=f"{label} ({average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color=PALETTE["text_muted"], ls="--", lw=0.8)
    a2.axhline(np.mean(y), color=PALETTE["text_muted"], ls="--", lw=0.8)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC{title_suffix}")
    a2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall{title_suffix}")
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    a2.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(results_path(fname))
    plt.close(fig)


def plot_reliability(plt, y, preds: dict, fname, n_bins=10):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color=PALETTE["text_muted"], ls="--", lw=0.8, label="perfect")
    hi = 0.0
    for (label, p), color in zip(preds.items(), SERIES_COLORS):
        frac, mean_p = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=4, lw=1.5, color=color,
                label=f"{label} (Brier {brier_score_loss(y, p):.4f})")
        hi = max(hi, mean_p.max(), frac.max())
    lim = min(1.0, hi * 1.1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted probability",
           ylabel="Observed readmission rate")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(results_path(fname))
    plt.close(fig)


def plot_hbar(plt, labels, values, fname, xlabel, top=20):
    labels, values = list(labels)[:top], list(values)[:top]
    fig, ax = plt.subplots(figsize=(5.5, 0.26 * len(labels) + 0.9))
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=PALETTE["series1"])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    fig.savefig(results_path(fname))
    plt.close(fig)
