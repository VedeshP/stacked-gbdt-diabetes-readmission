# stacked-gbdt-diabetes-readmission
Calibrated Stacked Gradient Boosting for 30-Day Readmission Prediction in Diabetic Patients

Retrospective study on the public, de-identified UCI "Diabetes 130-US Hospitals 1999-2008" dataset. Not for clinical use. See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for the full problem statement, architecture and methodology.

## Layout

```
code/      numbered pipeline scripts (01_eda.py, 02_preprocess.py, ...) + common.py helpers
tools/     kaggle_nb.py: builds Kaggle notebooks from code/ and pushes/runs/fetches them
results/   metrics (CSV/JSON) and 300 dpi figures; the single source of truth for the paper
paper/     draft.md, references.md
kaggle_dataset/  dataset-metadata.json for the Kaggle copy of the data
```

## Data

Download from UCI (https://archive.ics.uci.edu/dataset/296, CC BY 4.0) and unzip into `data/raw/` (git-ignored):

```bash
mkdir -p data/raw && curl -L -o data/raw/uci_296.zip "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip" && unzip -o data/raw/uci_296.zip -d data/raw
```

On Kaggle the scripts find the data automatically under `/kaggle/input` (private dataset `vedeshp/diabetes-130-us-hospitals-uci-296`). Override with `READMISSION_DATA_DIR`.

## Run locally

```bash
pip install -r requirements.txt
python code/01_eda.py         # EDA -> results/eda.json + figures
python code/02_preprocess.py  # features + patient-level split -> data/processed/
python code/03_baselines.py   # Optuna-tuned LR/RF/XGB/LGBM/CatBoost + class-weight ablation
python code/04_stacking.py    # grouped OOF stacking vs average vs singles, calibration, thresholds
python code/05_explain.py     # SHAP for base models and the stack
```

Quick smoke test (15% of patients, tiny budgets, writes to `results_fast/`): set `FAST=1` before running the scripts.

Run settings are environment variables (defaults in `code/common.py`): `N_TRIALS` (40), `TUNE_TIMEOUT` seconds per model (900), `TUNE_FOLDS` (3), `CV_FOLDS` (5), `N_SEEDS` (3), `N_BOOT` (1000), `USE_GPU` (auto), `EXCLUDE_EXPIRED` (1).

## Run on Kaggle

Requires the Kaggle CLI with `~/.kaggle/kaggle.json`. The notebook is generated from `code/`, so never edit it on Kaggle.

```bash
python tools/kaggle_nb.py build 01                 # just build build/kaggle/<slug>/notebook.ipynb
python tools/kaggle_nb.py push 01 --wait           # push, run, wait, download results into results/
python tools/kaggle_nb.py push 02 03 04 --gpu      # several scripts in one kernel, GPU on
python tools/kaggle_nb.py push --all --gpu --wait  # full pipeline
python tools/kaggle_nb.py status 01
python tools/kaggle_nb.py fetch 01 --sync          # download outputs, copy results/ into the repo
python tools/kaggle_nb.py push 02 04 05 --include results/best_params.json  # reuse tuned params
python tools/kaggle_nb.py push --all --env N_TRIALS=80 N_SEEDS=5            # change budgets
```

Each kernel starts from a clean machine, so include any earlier script whose outputs a later one needs. Settings (Kaggle username, dataset id, slug prefix) live in `tools/kaggle_config.json`.
