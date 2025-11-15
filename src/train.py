# src/train.py
"""
Training script that uses the FeatureEngineer in-pipeline.
Saves:
 - tuned model pipelines (best_rf_pipeline.joblib, best_xgb_pipeline.joblib)
 - ensemble pipelines (stack_pipeline.joblib, vote_pipeline.joblib)
 - feature_meta.joblib describing RAW features (used by the app)
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import xgboost as xgb

from src.data import load_data, quick_check
from src.features import clean_basic, FeatureEngineer, ENGINEERED_FEATURES

RND = 42
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

def prepare_data():
    df = load_data()
    quick_check(df)
    # minimal cleaning only
    df = clean_basic(df)
    # map target to 0/1
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes':1, 'No':0})
    # drop rows with missing target
    df = df[df['churn'].notnull()].copy()
    X_raw = df.drop(columns=['churn'])
    y = df['churn'].astype(int)
    return X_raw, y

def save_feature_meta_raw(X_train: pd.DataFrame):
    """
    Save metadata about RAW features, so the app can render simple inputs.
    meta contains:
      - raw_features: ordered list
      - dtypes: numeric|categorical
      - categories: unique values for categorical
      - numeric_defaults: median for numeric
      - engineered_features: list (so app knows what will be computed internally)
    """
    meta = {}
    raw_features = list(X_train.columns)
    meta['raw_features'] = raw_features

    dtypes = {}
    categories = {}
    numeric_defaults = {}

    for col in raw_features:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            dtypes[col] = 'numeric'
            numeric_defaults[col] = float(X_train[col].median() if X_train[col].notnull().any() else 0.0)
            categories[col] = []
        else:
            dtypes[col] = 'categorical'
            unique_vals = X_train[col].dropna().unique().tolist()
            unique_vals = [str(v) for v in unique_vals]
            categories[col] = unique_vals
            numeric_defaults[col] = None

    meta['dtypes'] = dtypes
    meta['categories'] = categories
    meta['numeric_defaults'] = numeric_defaults
    meta['engineered_features'] = ENGINEERED_FEATURES

    joblib.dump(meta, OUTPUT_DIR / "feature_meta.joblib")
    print("Saved raw feature_meta to", OUTPUT_DIR / "feature_meta.joblib")

def build_preprocessor_sample(X_feat: pd.DataFrame):
    """
    Build ColumnTransformer by inspecting a sample dataframe that already contains engineered features
    (this function is called after FeatureEngineer.transform on a sample to detect types).
    """
    num_cols = X_feat.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_feat.select_dtypes(exclude=['number']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], verbose=False)
    return preprocessor

def train_and_tune(X_train_raw, y_train, feat_engineer: FeatureEngineer):
    # generate a sample of engineered dataframe to know final columns
    X_train_feat = feat_engineer.transform(X_train_raw.copy())

    # build preprocessor based on engineered sample
    preproc = build_preprocessor_sample(X_train_feat)

    # full pipelines
    rf = RandomForestClassifier(random_state=RND, n_jobs=-1)
    rf_pipe = Pipeline([('feat', feat_engineer), ('pre', preproc), ('clf', rf)])

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RND, n_jobs=-1)
    xgb_pipe = Pipeline([('feat', feat_engineer), ('pre', preproc), ('clf', xgb_clf)])

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RND)

    # baseline CV
    print("Baseline CV (RF/XGB)...")
    rf_score = cross_val_score(rf_pipe, X_train_raw, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    xgb_score = cross_val_score(xgb_pipe, X_train_raw, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print("RF baseline ROC AUC:", rf_score.mean())
    print("XGB baseline ROC AUC:", xgb_score.mean())

    # Randomized search RF
    param_dist_rf = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [None, 6, 12],
        'clf__min_samples_leaf': [1, 2, 4]
    }
    rs_rf = RandomizedSearchCV(rf_pipe, param_dist_rf, n_iter=6, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=RND, verbose=1)
    rs_rf.fit(X_train_raw, y_train)
    best_rf = rs_rf.best_estimator_
    print("Best RF ROC AUC:", rs_rf.best_score_)
    print("Best RF params:", rs_rf.best_params_)

    # Randomized search XGB
    param_dist_xgb = {
        'clf__n_estimators': [100,200,400],
        'clf__max_depth': [3,5,7],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.6, 0.8, 1.0]
    }
    rs_xgb = RandomizedSearchCV(xgb_pipe, param_dist_xgb, n_iter=8, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=RND, verbose=1)
    rs_xgb.fit(X_train_raw, y_train)
    best_xgb = rs_xgb.best_estimator_
    print("Best XGB ROC AUC:", rs_xgb.best_score_)
    print("Best XGB params:", rs_xgb.best_params_)

    return best_rf, best_xgb, preproc

def build_ensemble(best_rf_pipe, best_xgb_pipe, feat_engineer: FeatureEngineer, preproc):
    rf_est = best_rf_pipe.named_steps['clf']
    xgb_est = best_xgb_pipe.named_steps['clf']

    stack = Pipeline([
        ('feat', feat_engineer),
        ('pre', preproc),
        ('stack', StackingClassifier(
            estimators=[('rf', rf_est), ('xgb', xgb_est)],
            final_estimator=RandomForestClassifier(n_estimators=200, random_state=RND),
            n_jobs=-1
        ))
    ])
    vote = Pipeline([
        ('feat', feat_engineer),
        ('pre', preproc),
        ('vote', VotingClassifier(estimators=[('rf', rf_est), ('xgb', xgb_est)], voting='soft'))
    ])
    return stack, vote

def evaluate_model(model, X_test_raw, y_test):
    y_pred = model.predict(X_test_raw)
    try:
        y_proba = model.predict_proba(X_test_raw)[:,1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None
    print("Accuracy:", accuracy_score(y_test, y_pred))
    if auc is not None:
        print("ROC AUC:", auc)
    print(classification_report(y_test, y_pred))

def main():
    X_raw, y = prepare_data()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=RND)

    # Save raw feature meta (so app knows which basic raw features to render)
    save_feature_meta_raw(X_train_raw)

    feat_engineer = FeatureEngineer()

    best_rf_pipe, best_xgb_pipe, preproc = train_and_tune(X_train_raw, y_train, feat_engineer)

    joblib.dump(best_rf_pipe, OUTPUT_DIR / "best_rf_pipeline.joblib")
    joblib.dump(best_xgb_pipe, OUTPUT_DIR / "best_xgb_pipeline.joblib")
    print("Saved tuned model pipelines to", OUTPUT_DIR)

    stack_pipe, vote_pipe = build_ensemble(best_rf_pipe, best_xgb_pipe, feat_engineer, preproc)
    stack_pipe.fit(X_train_raw, y_train)
    vote_pipe.fit(X_train_raw, y_train)
    joblib.dump(stack_pipe, OUTPUT_DIR / "stack_pipeline.joblib")
    joblib.dump(vote_pipe, OUTPUT_DIR / "vote_pipeline.joblib")
    print("Saved ensemble models.")

    print("=== Evaluating best RF on test ===")
    evaluate_model(best_rf_pipe, X_test_raw, y_test)
    print("=== Evaluating best XGB on test ===")
    evaluate_model(best_xgb_pipe, X_test_raw, y_test)
    print("=== Evaluating stacking on test ===")
    evaluate_model(stack_pipe, X_test_raw, y_test)
    print("=== Evaluating voting on test ===")
    evaluate_model(vote_pipe, X_test_raw, y_test)

if __name__ == "__main__":
    main()
