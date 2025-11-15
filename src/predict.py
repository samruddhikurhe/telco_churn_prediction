# src/predict.py
import joblib
from pathlib import Path
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_xgb_pipeline.joblib"

def load_model(path: str = None):
    p = MODEL_PATH if path is None else Path(path)
    model = joblib.load(p)
    return model

def predict_proba(df: pd.DataFrame, model=None):
    if model is None:
        model = load_model()
    return model.predict_proba(df)[:,1]

def predict_class(df: pd.DataFrame, model=None, threshold: float = 0.5):
    proba = predict_proba(df, model=model)
    return (proba >= threshold).astype(int)
