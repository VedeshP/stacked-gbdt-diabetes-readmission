"""Stateless cleaning + feature construction, and the sklearn preprocessors.

Everything here that is *stateless* (row-wise mappings) runs once in
02_preprocess.py. Everything *fitted* (encoders, scalers) lives in the sklearn
ColumnTransformers below, so it is re-fit inside every CV fold.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import NUMERIC_COLS

MISSING = "Missing"

# Diabetes drug columns. Near-constant ones (top value share >= 0.99 in
# results/eda.json -> near_constant_columns) are dropped.
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
NEAR_CONSTANT_DRUGS = [
    "glyburide-metformin", "nateglinide", "chlorpropamide", "acarbose", "miglitol",
    "tolazamide", "acetohexamide", "tolbutamide", "troglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "glipizide-metformin",
    "glimepiride-pioglitazone", "citoglipton", "examide",
]
KEPT_DRUGS = [d for d in DRUG_COLS if d not in NEAR_CONSTANT_DRUGS]

# Codes meaning "unknown" in IDS_mapping.csv, merged into one level.
UNKNOWN_IDS = {
    "admission_type_id": {5, 6, 8},
    "discharge_disposition_id": {18, 25, 26},
    "admission_source_id": {9, 15, 17, 20, 21},
}

DROP_COLS = ["weight"]  # ~97% missing


def icd9_group(code) -> str:
    """Map an ICD-9 code to the clinical groups used by Strack et al. (2014)."""
    if pd.isna(code):
        return MISSING
    code = str(code)
    if code[0] in "VE":
        return "Other"
    v = float(code)
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature construction. Returns features only (no ids, no target)."""
    out = pd.DataFrame(index=df.index)

    # demographics
    out["race"] = df["race"].fillna(MISSING)
    out["gender"] = df["gender"]
    out["age"] = df["age"].str.extract(r"\[(\d+)-")[0].astype(int) + 5  # bin midpoint

    # admission context (IDs as categorical strings, unknown codes merged)
    for col, unknown in UNKNOWN_IDS.items():
        out[col] = df[col].where(~df[col].isin(unknown), -1).astype(str).replace("-1", "unknown")
    out["payer_code"] = df["payer_code"].fillna(MISSING)
    out["medical_specialty"] = df["medical_specialty"].fillna(MISSING)

    # numeric utilisation
    for c in NUMERIC_COLS:
        out[c] = df[c].astype(float)
    out["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]).astype(float)

    # diagnoses -> clinical groups
    for c in ["diag_1", "diag_2", "diag_3"]:
        out[f"{c}_group"] = df[c].map(icd9_group)

    # labs ('None' = not performed is a real level)
    out["A1Cresult"] = df["A1Cresult"]
    out["max_glu_serum"] = df["max_glu_serum"]

    # medications
    for d in KEPT_DRUGS:
        out[d] = df[d]
    drugs = df[DRUG_COLS]
    out["n_drugs_active"] = (drugs != "No").sum(axis=1).astype(float)
    out["n_drugs_up"] = (drugs == "Up").sum(axis=1).astype(float)
    out["n_drugs_down"] = (drugs == "Down").sum(axis=1).astype(float)
    out["n_drugs_changed"] = out["n_drugs_up"] + out["n_drugs_down"]
    out["change"] = df["change"]
    out["diabetesMed"] = df["diabetesMed"]
    return out


def split_columns(X: pd.DataFrame):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


# ------------------------------------------------------------------ preprocessors

def onehot_preprocessor(num_cols, cat_cols, scale=False, min_frequency=50):
    """One-hot for categoricals (rare levels pooled), numerics passed through or scaled; dense."""
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_frequency, sparse_output=True), cat_cols),
        ],
        remainder="drop",
        # Dense output: XGBoost treats implicit zeros of a sparse matrix as
        # *missing*, which would silently turn numeric zeros into NaN.
        sparse_threshold=0.0,
    )


def native_preprocessor(num_cols, cat_cols):
    """Column selection only, pandas output, for models with native categorical handling."""
    return ColumnTransformer(
        [("num", "passthrough", num_cols), ("cat", "passthrough", cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def original_feature(name: str, input_cols) -> str:
    """Map a transformed feature name (e.g. 'cat__diag_1_group_Circulatory') to its source column."""
    base = name.split("__", 1)[-1]
    hits = [c for c in input_cols if base == c or base.startswith(c + "_")]
    return max(hits, key=len) if hits else base
