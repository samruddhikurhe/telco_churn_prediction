# src/features.py
"""
Feature engineering module for Telco churn project.

- FeatureEngineer: sklearn-compatible transformer that computes engineered features
  but respects user-provided values: it only computes a derived column if that column
  is missing or all-null in the input.
- clean_basic: minimal raw input cleaning (keeps raw features).
- ENGINEERED_FEATURES: list of engineered column names produced by FeatureEngineer.
"""

from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# List of engineered features produced by FeatureEngineer
ENGINEERED_FEATURES = [
    "tenure_group",
    "lifetime_value",
    "avg_charge_per_month"
]

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning applied to raw dataframe.
    - Removes customerid if present.
    - Coerces totalcharges to numeric.
    - Normalizes 'No internet service' / 'No phone service' strings to 'No'.
    """
    df = df.copy()
    if 'customerid' in df.columns:
        df.drop(columns=['customerid'], inplace=True)

    # totalcharges sometimes has spaces -- coerce to numeric
    if 'totalcharges' in df.columns:
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

    replace_vals = ['No internet service', 'No phone service', 'no internet service', 'no phone service']
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(replace_vals, 'No')

    return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer that augments raw telco data with derived features.
    Behavior change: *respects user-provided values* — if an engineered column
    is present (and not all NaN) in the input DataFrame, FeatureEngineer will NOT
    overwrite it. It computes engineered columns only when missing or fully-null.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        # stateless
        return self

    def _col_missing_or_all_null(self, df: pd.DataFrame, col: str) -> bool:
        """
        Return True if column is missing from df or all values are null/NaN.
        """
        if col not in df.columns:
            return True
        # consider values that are empty strings as null as well
        ser = df[col]
        # If all are null or empty strings
        if ser.isnull().all():
            return True
        # If values are strings and all empty/whitespace
        if ser.dtype == object and (ser.astype(str).str.strip() == "").all():
            return True
        return False

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Drop customerid if present
        if 'customerid' in df.columns:
            df = df.drop(columns=['customerid'])

        # totalcharges to numeric if present
        if 'totalcharges' in df.columns:
            df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

        # normalize some "no service" variants
        replace_vals = ['No internet service', 'No phone service', 'no internet service', 'no phone service']
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].replace(replace_vals, 'No')

        # seniorcitizen: if numeric 0/1 convert to 'No'/'Yes'
        if 'seniorcitizen' in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df['seniorcitizen']):
                    df['seniorcitizen'] = df['seniorcitizen'].map({0: 'No', 1: 'Yes'}).fillna(df['seniorcitizen'])
            except Exception:
                pass

        # ENGINEERED: tenure_group (only compute if missing or all null)
        if self._col_missing_or_all_null(df, 'tenure_group') and 'tenure' in df.columns:
            try:
                df['tenure_group'] = pd.cut(
                    df['tenure'],
                    bins=[-1, 6, 12, 24, 48, 60, 1000],
                    labels=['0-6', '7-12', '13-24', '25-48', '49-60', '61+']
                )
            except Exception:
                # if tenure is not numeric, coerce and try again
                try:
                    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
                    df['tenure_group'] = pd.cut(
                        df['tenure'],
                        bins=[-1, 6, 12, 24, 48, 60, 1000],
                        labels=['0-6', '7-12', '13-24', '25-48', '49-60', '61+']
                    )
                except Exception:
                    pass

        # ENGINEERED: lifetime_value = monthlycharges * tenure (compute only if missing)
        if self._col_missing_or_all_null(df, 'lifetime_value') and 'monthlycharges' in df.columns and 'tenure' in df.columns:
            # ensure numeric
            df['monthlycharges'] = pd.to_numeric(df['monthlycharges'], errors='coerce')
            try:
                df['lifetime_value'] = df['monthlycharges'] * df['tenure']
            except Exception:
                # if tenure not numeric, coerce
                try:
                    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
                    df['lifetime_value'] = df['monthlycharges'] * df['tenure']
                except Exception:
                    pass

        # ENGINEERED: avg_charge_per_month = totalcharges / tenure (fallback to monthlycharges)
        if self._col_missing_or_all_null(df, 'avg_charge_per_month') and 'tenure' in df.columns:
            # prefer totalcharges if available
            if 'totalcharges' in df.columns:
                try:
                    df['avg_charge_per_month'] = df['totalcharges'] / df['tenure'].replace(0, np.nan)
                except Exception:
                    try:
                        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
                        df['avg_charge_per_month'] = df['totalcharges'] / df['tenure'].replace(0, np.nan)
                    except Exception:
                        df['avg_charge_per_month'] = np.nan
            # fallback to monthlycharges
            if 'avg_charge_per_month' not in df.columns or df['avg_charge_per_month'].isnull().all():
                if 'monthlycharges' in df.columns:
                    df['monthlycharges'] = pd.to_numeric(df['monthlycharges'], errors='coerce')
                    df['avg_charge_per_month'] = df['monthlycharges']

        # ensure engineered columns exist (may be NaN)
        for col in ENGINEERED_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        return df

# Backwards-compatible function
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer()
    return fe.transform(df)
