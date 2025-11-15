# app/app.py
"""
Streamlit app: ask only friendly basic inputs, advanced inputs are optional.
If advanced fields are left blank the pipeline computes them; if the user supplies
advanced/raw values the model will use them (FeatureEngineer respects supplied values).
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import sys

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = MODELS_DIR / "best_xgb_pipeline.joblib"
FEATURE_META_FILE = MODELS_DIR / "feature_meta.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    feature_meta = joblib.load(FEATURE_META_FILE)
    return model, feature_meta

try:
    model, feature_meta = load_artifacts()
except Exception as e:
    # Show a helpful message in Streamlit if running via Streamlit
    try:
        st.error(f"Could not load model or feature metadata: {e}")
        st.write("Make sure you have run the training step (python -m src.train) and that files exist in the models/ folder.")
        st.stop()
    except Exception:
        # If st.* is not available (running as `python app.py`), print and exit
        print(f"Could not load model or feature metadata: {e}", file=sys.stderr)
        sys.exit(1)

# Polished title and description
st.title("Telco Churn — Churn Probability Predictor")
st.markdown(
    """
    Predict the probability that a telecom customer will churn (leave the service).
    Provide a few basic customer details below and press **Predict** — the model will
    compute internal advanced features (like lifetime value or average charge per month)
    automatically where they are missing.  

    **How it works**
    - The app asks for a small set of user-friendly inputs (gender, senior status, tenure, contract, etc.).
    - Advanced fields are available under *Advanced inputs* — leave them blank for automatic computation, or provide a custom value to override.
    - The saved prediction pipeline will **use user-provided values** for any advanced fields you supply, otherwise it computes them internally.
    """
)

# Raw features metadata
raw_features = feature_meta['raw_features']
dtypes = feature_meta['dtypes']
categories = feature_meta['categories']
numeric_defaults = feature_meta['numeric_defaults']
engineered = feature_meta.get('engineered_features', [])

# Choose a compact set of main fields to show by default
main_fields = ['gender', 'seniorcitizen', 'partner', 'dependents', 'tenure',
               'phoneservice', 'internetservice', 'contract', 'monthlycharges', 'paymentmethod']

# Keep only those present in raw_features (safe)
main_fields = [f for f in main_fields if f in raw_features]

with st.form("raw_form"):
    inputs = {}

    # Render main fields in two columns
    cols = st.columns(2)
    for i, col in enumerate(main_fields):
        target = cols[i % 2]
        with target:
            if col == 'seniorcitizen':
                # render as explicit Yes/No choice (user asked this specifically)
                val = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
                inputs[col] = val
                continue

            if dtypes.get(col) == 'numeric':
                default = numeric_defaults.get(col, 0.0) or 0.0
                if float(default).is_integer():
                    inputs[col] = st.number_input(col.replace("_", " ").title(), value=int(default))
                else:
                    inputs[col] = st.number_input(col.replace("_", " ").title(), value=float(default))
            else:
                opts = categories.get(col, [])
                if not opts:
                    opts = ["Unknown"]
                opts = [str(o) for o in opts]
                inputs[col] = st.selectbox(col.replace("_", " ").title(), opts, index=0)

    # Advanced inputs: remaining raw features + any raw feature that user might want to override
    advanced_raw = [c for c in raw_features if c not in main_fields]

    if advanced_raw or engineered:
        with st.expander("Advanced inputs (optional — leave blank for automatic computation)"):
            st.write("Provide values here only if you want to override the model's own computed values.")
            # Create an inputs area for advanced raw features
            for col in advanced_raw:
                # For categorical advanced fields, include an 'Auto' option at the top
                if dtypes.get(col) == 'numeric':
                    # show a checkbox toggling whether the user wants to supply this numeric value
                    supply = st.checkbox(f"Provide custom value for '{col}'?", key=f"chk_{col}")
                    if supply:
                        default = numeric_defaults.get(col, 0.0) or 0.0
                        if float(default).is_integer():
                            inputs[col] = st.number_input(col.replace("_", " ").title(), value=int(default), key=f"num_{col}")
                        else:
                            inputs[col] = st.number_input(col.replace("_", " ").title(), value=float(default), key=f"num_{col}")
                    else:
                        inputs[col] = None  # None => let pipeline compute / handle
                else:
                    opts = categories.get(col, [])
                    if not opts:
                        opts = []
                    # prepend Auto option
                    opts_display = ["Auto (let model compute)"] + [str(o) for o in opts]
                    choice = st.selectbox(col.replace("_", " ").title(), opts_display, index=0, key=f"adv_{col}")
                    if choice == "Auto (let model compute)":
                        inputs[col] = None
                    else:
                        inputs[col] = choice

            # Also allow user to optionally supply engineered features (they are not required)
            st.markdown("---")
            st.write("Engineered features (optional). If you leave these blank, the model will calculate them.")
            for col in engineered:
                # engineered features are typically numeric or categorical (e.g., tenure_group)
                if col in raw_features:
                    # if somehow an engineered column is also raw, it was handled above
                    continue
                if col in dtypes and dtypes.get(col) == 'numeric':
                    supply = st.checkbox(f"Provide custom engineered value for '{col}'?", key=f"chk_eng_{col}")
                    if supply:
                        default = numeric_defaults.get(col, 0.0) or 0.0
                        if float(default).is_integer():
                            inputs[col] = st.number_input(col.replace("_", " ").title(), value=int(default), key=f"eng_num_{col}")
                        else:
                            inputs[col] = st.number_input(col.replace("_", " ").title(), value=float(default), key=f"eng_num_{col}")
                    else:
                        inputs[col] = None
                else:
                    # categorical engineered (e.g., tenure_group)
                    opts = feature_meta.get('categories', {}).get(col, [])
                    opts_display = ["Auto (let model compute)"] + (opts if opts else [])
                    choice = st.selectbox(col.replace("_", " ").title(), opts_display, index=0, key=f"eng_{col}")
                    inputs[col] = None if choice == "Auto (let model compute)" else choice

    submitted = st.form_submit_button("Predict churn probability")

if submitted:
    # Build the input DataFrame in exact order of raw_features (the model expects raw_features)
    row = {}
    for col in raw_features:
        val = inputs.get(col, None)
        # Interpret None / empty as np.nan so pipeline imputers and FeatureEngineer can act
        if val is None:
            row[col] = np.nan
        else:
            row[col] = val

    input_df = pd.DataFrame([row], columns=raw_features)

    # Convert numeric columns to numeric types
    for col in raw_features:
        if dtypes.get(col) == 'numeric':
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Now pass raw_df to the pipeline (the pipeline contains FeatureEngineer)
    try:
        proba = model.predict_proba(input_df)[0, 1]
        st.success(f"Predicted churn probability: **{proba:.3f}**")
        st.write("Tip: use Advanced inputs only if you want to override the model's automatic calculations.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Input dataframe (first row):")
        st.write(input_df.head(1))
