# src/data.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"

def load_data(path: str = None) -> pd.DataFrame:
    p = DATA_PATH if path is None else Path(path)
    df = pd.read_csv(p)
    # standardize column names to lower-case and underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def quick_check(df):
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing percent:\n", (df.isnull().mean() * 100).sort_values(ascending=False).head(10))
    print("Target distribution:\n", df['churn'].value_counts(normalize=True))
