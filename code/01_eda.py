"""01 - Exploratory data analysis.

Outputs (results/):
  eda.json                               headline numbers quoted in the paper
  eda_missingness.csv                    per-column missing fraction
  eda_cardinality.csv                    per-categorical cardinality + top-value share
  eda_numeric_summary.csv                describe() of numeric columns
  eda_rate_by_<feature>.csv              30-day readmission rate per level (Wilson 95% CI)
  eda_class_balance.png
  eda_missingness.png
  eda_readmit_rate_by_age.png
  eda_readmit_rate_by_number_inpatient.png
  eda_readmit_rate_by_discharge_disposition.png
"""
import numpy as np
import pandas as pd

from common import (
    CODE_ID_COLS, EXPIRED_IDS, HOSPICE_IDS, ID_COLS, NUMERIC_COLS, PALETTE,
    TARGET, TARGET_RAW, find_data_dir, load_raw, results_path, save_json,
    set_seed, setup_matplotlib,
)

LAB_COLS = ["A1Cresult", "max_glu_serum"]
NEAR_CONSTANT_SHARE = 0.99


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def rate_table(df, col, max_level=None):
    s = df[col]
    if max_level is not None:
        s = s.clip(upper=max_level)
    g = df.groupby(s, dropna=False)[TARGET].agg(["sum", "count"])
    g.columns = ["n_pos", "n"]
    g["rate"] = g["n_pos"] / g["n"]
    lo, hi = wilson_ci(g["n_pos"].to_numpy(), g["n"].to_numpy())
    g["ci_low"], g["ci_high"] = lo, hi
    g.index.name = col
    return g.reset_index()


def plot_rate(plt, table, col, overall, fname, xlabel, horizontal=False, labels=None):
    labels = labels if labels is not None else table[col].astype(str)
    err = np.vstack([table["rate"] - table["ci_low"], table["ci_high"] - table["rate"]])
    if horizontal:
        fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(table) + 1.0))
        y = np.arange(len(table))
        ax.barh(y, table["rate"], height=0.6, color=PALETTE["series1"],
                xerr=err, error_kw=dict(ecolor=PALETTE["text_muted"], lw=0.8, capsize=2))
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.axvline(overall, color=PALETTE["text_muted"], ls="--", lw=1,
                   label=f"overall = {overall:.3f}")
        ax.set_xlabel("30-day readmission rate (95% Wilson CI)")
        ax.set_ylabel(xlabel)
        ax.grid(axis="y", visible=False)
    else:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        x = np.arange(len(table))
        ax.bar(x, table["rate"], width=0.6, color=PALETTE["series1"],
               yerr=err, error_kw=dict(ecolor=PALETTE["text_muted"], lw=0.8, capsize=2))
        ax.set_xticks(x, labels)
        ax.axhline(overall, color=PALETTE["text_muted"], ls="--", lw=1,
                   label=f"overall = {overall:.3f}")
        ax.set_ylabel("30-day readmission rate (95% Wilson CI)")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.savefig(results_path(fname))
    plt.close(fig)


def main():
    set_seed()
    plt = setup_matplotlib()
    df = load_raw()
    n = len(df)

    # ---- target -------------------------------------------------------
    raw_counts = df[TARGET_RAW].value_counts().to_dict()
    n_pos = int(df[TARGET].sum())
    dead_or_hospice = df["discharge_disposition_id"].isin(EXPIRED_IDS + HOSPICE_IDS)
    df_alive = df[~dead_or_hospice]

    # ---- patients -----------------------------------------------------
    enc_per_patient = df.groupby("patient_nbr").size()
    first_enc = df.sort_values("encounter_id").drop_duplicates("patient_nbr")

    # ---- missingness ('?' -> NaN) ------------------------------------
    miss = df.drop(columns=[TARGET]).isna().mean().sort_values(ascending=False)
    miss_df = miss.rename("missing_fraction").rename_axis("column").reset_index()
    miss_df.to_csv(results_path("eda_missingness.csv"), index=False)

    lab_not_performed = {c: float((df[c] == "None").mean()) for c in LAB_COLS}

    # ---- cardinality of categoricals ---------------------------------
    cat_cols = [c for c in df.columns
                if c not in ID_COLS + NUMERIC_COLS + [TARGET, TARGET_RAW]]
    card_rows = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False, normalize=True)
        card_rows.append({
            "column": c,
            "n_unique": int(df[c].nunique(dropna=True)),
            "top_value": str(vc.index[0]),
            "top_share": float(vc.iloc[0]),
            "missing_fraction": float(df[c].isna().mean()),
        })
    card_df = pd.DataFrame(card_rows).sort_values("n_unique", ascending=False)
    card_df.to_csv(results_path("eda_cardinality.csv"), index=False)
    near_constant = card_df.loc[card_df["top_share"] >= NEAR_CONSTANT_SHARE, "column"].tolist()

    # ---- numeric summary ---------------------------------------------
    df[NUMERIC_COLS].describe().T.to_csv(results_path("eda_numeric_summary.csv"))

    # ---- readmission rate by key features ----------------------------
    overall = n_pos / n
    by_age = rate_table(df, "age")
    by_inp = rate_table(df, "number_inpatient", max_level=5)
    by_dis = rate_table(df, "discharge_disposition_id")
    for name, t in [("age", by_age), ("number_inpatient", by_inp),
                    ("discharge_disposition_id", by_dis)]:
        t.to_csv(results_path(f"eda_rate_by_{name}.csv"), index=False)

    # ---- summary json -------------------------------------------------
    summary = {
        "data_dir": str(find_data_dir()),
        "n_encounters": n,
        "n_columns_raw": int(df.shape[1] - 1),  # excluding derived target
        "n_patients": int(df["patient_nbr"].nunique()),
        "duplicate_encounter_ids": int(df["encounter_id"].duplicated().sum()),
        "encounters_per_patient": {
            "mean": float(enc_per_patient.mean()),
            "median": float(enc_per_patient.median()),
            "max": int(enc_per_patient.max()),
            "share_patients_with_multiple": float((enc_per_patient > 1).mean()),
            "share_encounters_from_repeat_patients": float(
                enc_per_patient[enc_per_patient > 1].sum() / n),
        },
        "target_raw_counts": raw_counts,
        "target_binary": {
            "definition": "readmitted == '<30' -> 1, else 0",
            "n_positive": n_pos,
            "n_negative": n - n_pos,
            "positive_rate": overall,
        },
        "expired_or_hospice": {
            "discharge_ids_expired": EXPIRED_IDS,
            "discharge_ids_hospice": HOSPICE_IDS,
            "n_encounters": int(dead_or_hospice.sum()),
            "n_positive_among_them": int(df.loc[dead_or_hospice, TARGET].sum()),
            "positive_rate_after_exclusion": float(df_alive[TARGET].mean()),
            "n_encounters_after_exclusion": int(len(df_alive)),
        },
        "first_encounter_per_patient": {
            "n": int(len(first_enc)),
            "positive_rate": float(first_enc[TARGET].mean()),
        },
        "missing_fraction_nonzero": {k: float(v) for k, v in miss.items() if v > 0},
        "lab_not_performed_fraction": lab_not_performed,
        "n_categorical_columns": len(cat_cols),
        "near_constant_columns": near_constant,
        "near_constant_threshold": NEAR_CONSTANT_SHARE,
        "icd9_unique": {c: int(df[c].nunique()) for c in ["diag_1", "diag_2", "diag_3"]},
    }
    save_json(summary, "eda.json")

    # ---- figures ------------------------------------------------------
    order = ["NO", ">30", "<30"]
    counts = [raw_counts.get(k, 0) for k in order]
    colors = [PALETTE["series1"], PALETTE["series1"], PALETTE["series2"]]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars = ax.bar(order, counts, width=0.6, color=colors)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c / n:.1%})",
                ha="center", va="bottom", fontsize=8, color=PALETTE["text"])
    ax.set_ylabel("Encounters")
    ax.set_xlabel("readmitted (orange = positive class, <30 days)")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="x", visible=False)
    fig.savefig(results_path("eda_class_balance.png"))
    plt.close(fig)

    m = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(m) + 0.8))
    y = np.arange(len(m))
    ax.barh(y, m.values, height=0.6, color=PALETTE["series1"])
    ax.set_yticks(y, m.index)
    ax.invert_yaxis()
    for yi, v in zip(y, m.values):
        ax.text(v + 0.01, yi, f"{v:.1%}" if v >= 0.01 else f"{v:.2%}", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Fraction missing ('?')")
    ax.grid(axis="y", visible=False)
    fig.savefig(results_path("eda_missingness.png"))
    plt.close(fig)

    age_labels = by_age["age"].str.strip("[)")
    plot_rate(plt, by_age, "age", overall, "eda_readmit_rate_by_age.png",
              "Age group (years)", labels=age_labels)
    inp_labels = [str(int(v)) if v < 5 else "5+" for v in by_inp["number_inpatient"]]
    plot_rate(plt, by_inp, "number_inpatient", overall,
              "eda_readmit_rate_by_number_inpatient.png",
              "Inpatient visits in prior year", labels=inp_labels)
    by_dis_big = by_dis[by_dis["n"] >= 100]
    plot_rate(plt, by_dis_big, "discharge_disposition_id", overall,
              "eda_readmit_rate_by_discharge_disposition.png",
              "Discharge disposition ID (n >= 100)", horizontal=True)

    print(f"encounters={n} patients={summary['n_patients']} "
          f"positive_rate={overall:.4f} near_constant={near_constant}")
    print(f"wrote results to {results_path('eda.json').parent}")


if __name__ == "__main__":
    main()
