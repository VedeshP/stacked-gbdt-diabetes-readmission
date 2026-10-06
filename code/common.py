"""Shared constants and helpers for all numbered scripts.

Runs unchanged locally and on Kaggle:
- Data dir: $READMISSION_DATA_DIR, else any /kaggle/input/**/diabetic_data.csv,
  else <repo>/data/raw.
- Results dir: <repo>/results (on Kaggle <repo> is /kaggle/working).
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RAW_FILE = "diabetic_data.csv"
IDS_FILE = "IDS_mapping.csv"

TARGET_RAW = "readmitted"
TARGET = "readmit_30d"

# discharge_disposition_id codes from IDS_mapping.csv
EXPIRED_IDS = [11, 19, 20, 21]
HOSPICE_IDS = [13, 14]

ID_COLS = ["encounter_id", "patient_nbr"]
CODE_ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_data_dir() -> Path:
    env = os.environ.get("READMISSION_DATA_DIR")
    if env:
        return Path(env)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = sorted(kaggle_input.rglob(RAW_FILE))
        if hits:
            return hits[0].parent
    return REPO_ROOT / "data" / "raw"


def load_raw() -> pd.DataFrame:
    """Load the encounter table.

    '?' is the dataset's missing marker. The literal string 'None' in
    A1Cresult / max_glu_serum means 'test not performed' and must stay a
    category, so pandas' default NA strings are disabled.
    """
    path = find_data_dir() / RAW_FILE
    df = pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)
    df[TARGET] = (df[TARGET_RAW] == "<30").astype(int)
    return df


def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def save_json(obj, name: str) -> Path:
    path = results_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


# Figure style shared by all scripts (300 dpi PNG, light background for print)
PALETTE = {
    "series1": "#2a78d6",  # blue
    "series2": "#eb6834",  # orange
    "series3": "#1baf7a",  # aqua
    "text": "#0b0b0b",
    "text_muted": "#52514e",
    "grid": "#e4e3df",
}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": PALETTE["text_muted"],
        "axes.labelcolor": PALETTE["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": PALETTE["text_muted"],
        "ytick.color": PALETTE["text_muted"],
    })
    return plt
