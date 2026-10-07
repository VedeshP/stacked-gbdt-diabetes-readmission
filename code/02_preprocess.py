"""02 - Cleaning, feature construction, and the patient-level train/test split.

- Excludes encounters discharged as expired or to hospice (cannot be readmitted);
  set EXCLUDE_EXPIRED=0 for the sensitivity analysis.
- Builds features row-wise (features.build_features). Fitted encoders are NOT
  applied here; they live in the model pipelines and are fit per CV fold.
- Splits by patient with StratifiedGroupKFold(5): fold 0 = held-out TEST
  (~20% of patients). No patient appears in both train and test (asserted).

Outputs:
  data/processed/dataset.joblib   X, y, groups, encounter_id, is_test, column lists
  results/split.json              sizes, positive rates, leakage check
  results/features.csv            feature list with type and number of levels
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import (EXPIRED_IDS, FAST, FAST_N_ROWS, HOSPICE_IDS, SEED, TARGET,
                    load_raw, results_path, save_dataset, save_json, set_seed)
from features import DROP_COLS, NEAR_CONSTANT_DRUGS, build_features, split_columns

EXCLUDE_EXPIRED = os.environ.get("EXCLUDE_EXPIRED", "1") == "1"
N_OUTER_FOLDS = 5


def main():
    set_seed()
    df = load_raw()
    n_raw = len(df)

    excluded = df["discharge_disposition_id"].isin(EXPIRED_IDS + HOSPICE_IDS)
    if EXCLUDE_EXPIRED:
        df = df[~excluded]
    if FAST:  # subsample whole patients for a quick smoke test
        rng = np.random.default_rng(SEED)
        patients = df["patient_nbr"].unique()
        rng.shuffle(patients)
        keep = patients[: int(len(patients) * FAST_N_ROWS / len(df))]
        df = df[df["patient_nbr"].isin(keep)]
    df = df.sort_values("encounter_id").reset_index(drop=True)

    X = build_features(df)
    y = df[TARGET].to_numpy()
    groups = df["patient_nbr"].to_numpy()

    outer = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    _, test_idx = next(outer.split(X, y, groups))
    is_test = np.zeros(len(df), dtype=bool)
    is_test[test_idx] = True

    shared = set(groups[is_test]) & set(groups[~is_test])
    assert not shared, f"patient leakage: {len(shared)} patients in both train and test"

    num_cols, cat_cols = split_columns(X)
    save_dataset({
        "X": X, "y": y, "groups": groups, "is_test": is_test,
        "encounter_id": df["encounter_id"].to_numpy(),
        "num_cols": num_cols, "cat_cols": cat_cols,
    })

    def part(mask):
        return {"n_encounters": int(mask.sum()),
                "n_patients": int(len(np.unique(groups[mask]))),
                "n_positive": int(y[mask].sum()),
                "positive_rate": float(y[mask].mean())}

    save_json({
        "fast_mode": FAST,
        "n_encounters_raw": n_raw,
        "exclude_expired_or_hospice": EXCLUDE_EXPIRED,
        "n_excluded_expired_or_hospice": int(excluded.sum()) if EXCLUDE_EXPIRED else 0,
        "n_encounters_used": int(len(df)),
        "split_method": f"StratifiedGroupKFold(n_splits={N_OUTER_FOLDS}, shuffle=True, "
                        f"random_state={SEED}) on patient_nbr; fold 0 = test",
        "train": part(~is_test),
        "test": part(is_test),
        "test_fraction_encounters": float(is_test.mean()),
        "patients_in_both_train_and_test": len(shared),
        "dropped_columns": ["encounter_id", "patient_nbr"] + DROP_COLS + NEAR_CONSTANT_DRUGS,
        "n_features": X.shape[1],
        "n_numeric_features": len(num_cols),
        "n_categorical_features": len(cat_cols),
    }, "split.json")

    pd.DataFrame({
        "feature": X.columns,
        "type": ["numeric" if c in num_cols else "categorical" for c in X.columns],
        "n_levels": [X[c].nunique() for c in X.columns],
    }).to_csv(results_path("features.csv"), index=False)

    print(f"used={len(df)} train={int((~is_test).sum())} test={int(is_test.sum())} "
          f"features={X.shape[1]} (num={len(num_cols)}, cat={len(cat_cols)}) leakage=0")


if __name__ == "__main__":
    main()
