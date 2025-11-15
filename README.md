# Telco Churn Prediction

**Project:** Telco Customer Churn prediction using ensemble methods (RandomForest, XGBoost) with deployment via Streamlit.

---

## Overview

This repository implements an end-to-end ML pipeline for predicting customer churn on a Telco dataset. The project includes:

* Data loading and minimal cleaning (`src/data.py`).
* Feature engineering implemented as a scikit-learn transformer (`src/features.py`).
* Training, hyperparameter tuning and model saving (`src/train.py`).
* Predict wrapper utilities (`src/predict.py`).
* A Streamlit demo app for interactive predictions (`app/app.py`).
* Dockerfile and `requirements.txt` for containerized deployment.

> **Note:** The `data/` and `models/` folders are intentionally ignored from version control. Place your dataset CSV in `data/telco_churn.csv` before training. After training, model artifacts will be saved into `models/`.

---

## Repository layout

```
lab10-vsproject/
├─ data/                    # place telco_churn.csv here (NOT committed)
├─ models/                  # saved model artifacts appear here (NOT committed)
├─ src/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ features.py
│  ├─ train.py
│  └─ predict.py
├─ app/
│  └─ app.py                # Streamlit demo
├─ Dockerfile
├─ requirements.txt
├─ README.md                # this file
└─ .gitignore
```

---

## Quick start — run locally (Windows)

These commands assume you are in the project root folder.

### 1) Create & activate virtual environment

**PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**CMD**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Git Bash** (if using Git Bash on Windows)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2) Install required packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Place dataset

Download the Telco Customer Churn CSV (e.g. `WA_Fn-UseC_-Telco-Customer-Churn.csv`) and save as:

```
data/telco_churn.csv
```

### 4) Train models (this creates `models/feature_meta.joblib` and model files)

Run from project root:

```bash
# preferred: use venv python explicitly
.venv/Scripts/python -m src.train
# on Unix/mac use: .venv/bin/python -m src.train
```

Training will perform basic preprocessing, feature engineering inside a transformer, hyperparameter search for RandomForest and XGBoost, and save pipelines under `models/`.

### 5) Run Streamlit app (demo)

Run with the venv python so it uses the same environment:

```bash
.venv/Scripts/python -m streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Quick start — run locally (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# place data/telco_churn.csv
.venv/bin/python -m src.train
.venv/bin/python -m streamlit run app/app.py
```

---

## Docker (optional)

Build and run container (project root):

```bash
docker build -t lab10-churn-app:latest .
docker run -p 8501:8501 lab10-churn-app:latest
```

Then open `http://localhost:8501`.

> If you do not want to bake `models/` into the image, you can mount the `models/` folder at runtime: `-v $(pwd)/models:/app/models`.

---

## Notes on running in different shells

* If you encounter `python: command not found` in Git Bash, either activate the venv or run the venv python directly: `.venv/Scripts/python -m ...`.
* If you see `sed: command not found` in Git Bash, add Git's Unix tools to PATH or reinstall Git for Windows. This warning is not fatal for the app but fixing PATH improves shell behavior.

---

## Git & GitHub

1. Initialize repo (if not already):

```bash
git init
git add .
git commit -m "Initial commit"
```

2. Create a repo on GitHub (web UI or `gh`), then push:

```bash
git remote add origin https://github.com/<YOUR-USERNAME>/<REPO>.git
git branch -M main
git push -u origin main
```

**Avoid committing** `data/` and `models/`. Use GitHub Releases or Git LFS for large model files if you want to share them.

---

## Reproducibility & troubleshooting

* If `joblib.load` fails with `ModuleNotFoundError: No module named 'src'`, run Streamlit from project root (so `src` is importable) or add the project root to `sys.path` at the top of `app/app.py`.

* If you see `ModuleNotFoundError` for other packages, ensure you installed `requirements.txt` in the same environment running Streamlit.

* To debug, run small checks:

```bash
# check python interpreter used
.venv/Scripts/python -c "import sys; print(sys.executable, sys.version)"
# list model files
ls -la models
```

---

## How the app behaves (UX summary)

* The Streamlit app asks the user for **basic, user-friendly raw features** (gender, senior status, tenure, contract, monthly charges, etc.).
* Advanced engineered fields (e.g., `lifetime_value`, `avg_charge_per_month`, `tenure_group`) are available under *Advanced inputs* and are optional.

  * If user leaves them blank, the pipeline's `FeatureEngineer` computes them automatically.
  * If the user provides them, the pipeline **respects** those values and uses them instead of computing.
* The app loads the saved pipeline (`models/best_xgb_pipeline.joblib` by default) and outputs the churn probability.

---

## Files to inspect

* `src/features.py` — FeatureEngineer transformer (computes advanced features)
* `src/train.py` — training, tuning, and artifact saving
* `app/app.py` — Streamlit UI

---
