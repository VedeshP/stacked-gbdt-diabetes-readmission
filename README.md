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
python code/01_eda.py
```

## Run on Kaggle

Requires the Kaggle CLI with `~/.kaggle/kaggle.json`. The notebook is generated from `code/`, so never edit it on Kaggle.

```bash
python tools/kaggle_nb.py build 01                 # just build build/kaggle/<slug>/notebook.ipynb
python tools/kaggle_nb.py push 01 --wait           # push, run, wait, download results into results/
python tools/kaggle_nb.py push 02 03 04 --gpu      # several scripts in one kernel, GPU on
python tools/kaggle_nb.py push --all --gpu --wait  # full pipeline
python tools/kaggle_nb.py status 01
python tools/kaggle_nb.py fetch 01 --sync          # download outputs, copy results/ into the repo
```

Each kernel starts from a clean machine, so include any earlier script whose outputs a later one needs. Settings (Kaggle username, dataset id, slug prefix) live in `tools/kaggle_config.json`.
