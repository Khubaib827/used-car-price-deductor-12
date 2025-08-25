#!/usr/bin/env python3
"""
Used Car Price Prediction App (2024 data → predict 2025 used car prices)

Dataset columns detected (from provided CSV):
- addref (ID) — will be dropped
- city, assembly, body, make, model, transmission, fuel, color, registered — categorical
- year, engine, mileage — numeric
- price — target (float)

What this script does:
1) Loads dataset from a local CSV path.
2) Light EDA & data quality checks (printed).
3) Preprocessing with a robust ColumnTransformer (median/mode imputation + one-hot).
4) Train/test split.
5) Trains 5 regression models + an auxiliary Logistic Regression classifier on binned prices:
   - LinearRegression
   - RidgeCV
   - RandomForestRegressor
   - ExtraTreesRegressor
   - HistGradientBoostingRegressor
   - (Aux) LogisticRegression on price bins, for completeness since spec required it.
6) Compares regression models (MAE, RMSE, R2) and selects the best by RMSE.
7) Saves metrics, best model, and shows example predictions.
8) Provides a simple predict() helper.

Run:
    python used_car_price_prediction.py --data /path/to/pakwheels_used_car_data_v02.csv

Outputs:
    - metrics_report.csv
    - best_model.joblib
    - feature_names.json
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

from joblib import dump

RANDOM_STATE = 42

def read_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    return df

def basic_eda(df: pd.DataFrame) -> None:
    print("=== BASIC EDA ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nNull counts:\n", df.isna().sum())
    print("\nSample rows:\n", df.sample(min(5, len(df)), random_state=RANDOM_STATE))

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    # Drop ID-like columns
    drop_cols = []
    if 'addref' in df.columns:
        drop_cols.append('addref')

    # Target column
    target = 'price'
    if target not in df.columns:
        raise KeyError("Expected target column 'price' not found in dataset.")

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in drop_cols + [target]]

    # Separate numerical & categorical
    num_cols = []
    cat_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Add an engineered 'age' feature if 'year' exists
    if 'year' in num_cols or 'year' in df.columns:
        # age relative to 2024 (training-year proxy)
        df['age'] = 2024 - df['year']
        if 'year' in num_cols:
            num_cols.append('age')
        else:
            # year might be object due to bad rows—handle below by coercion in pipeline
            num_cols.append('age')

    # Numeric pipeline: median impute
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical pipeline: most_frequent impute + OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Recompute feature_cols to include engineered features
    feature_cols = list(set(num_cols + cat_cols))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )

    return preprocessor, feature_cols

def get_models() -> Dict[str, Any]:
    models = {
        'LinearRegression': LinearRegression(),
        'RidgeCV': RidgeCV(alphas=np.logspace(-3, 3, 13)),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        'ExtraTreesRegressor': ExtraTreesRegressor(
            n_estimators=600, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(
            learning_rate=0.07, max_depth=None, max_iter=500, random_state=RANDOM_STATE
        ),
    }
    return models

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def train_and_compare(df: pd.DataFrame) -> Dict[str, Any]:
    # Build preprocessor
    preprocessor, feature_cols = build_preprocessor(df)

    # Prepare data
    X = df[feature_cols].copy()
    y = df['price'].astype(float)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train regression models
    models = get_models()
    metrics = {}
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics[name] = evaluate_model(y_test, preds)
        fitted[name] = pipe
        print(f"[{name}] -> {metrics[name]}")

    # Select best by RMSE (lower is better)
    best_name = min(metrics.keys(), key=lambda k: metrics[k]['RMSE'])
    best_pipe = fitted[best_name]

    # Auxiliary: Logistic Regression on binned prices (for spec)
    try:
        bins = np.quantile(y_train, [0, 0.25, 0.5, 0.75, 1.0])
        bins[0] -= 1e-6  # ensure left-inclusivity
        price_bin = pd.cut(y, bins=bins, labels=['Q1','Q2','Q3','Q4'])
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, price_bin, test_size=0.2, random_state=RANDOM_STATE, stratify=price_bin
        )
        clf = Pipeline(steps=[('preprocess', preprocessor),
                              ('clf', LogisticRegression(max_iter=1000))])
        clf.fit(Xc_train, yc_train)
        cls_acc = clf.score(Xc_test, yc_test)
        print(f"(Aux) LogisticRegression price-quantile accuracy: {cls_acc:.4f}")
    except Exception as e:
        print("(Aux) LogisticRegression step skipped due to:", repr(e))

    # Evaluate best on a few samples for illustration
    head_preds = best_pipe.predict(X_test[:10])
    for i, (p, a) in enumerate(zip(head_preds, y_test.iloc[:10])):
        print(f"Sample {i}: Predicted={p:,.0f} | Actual={a:,.0f}")

    # Persist outputs
    metrics_df = pd.DataFrame(metrics).T.sort_values('RMSE')
    metrics_df.to_csv('metrics_report.csv', index=True)
    dump(best_pipe, 'best_model.joblib')

    # Save feature names after preprocessing for reference
    # Fit a one-step pipeline to extract feature names
    preprocessor.fit(X_train)
    # numeric + ohe names
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    # Post-OHE names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_names = ohe.get_feature_names_out(cat_features).tolist()
    feature_names = list(num_features) + ohe_names
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    results = {
        'best_model': best_name,
        'metrics': metrics_df.to_dict(orient='index'),
        'samples': [{'pred': float(p), 'actual': float(a)} for p, a in zip(head_preds, y_test.iloc[:10])]
    }
    return results

def predict(samples: pd.DataFrame, model_path: str = 'best_model.joblib') -> np.ndarray:
    """Load the persisted best model and predict for new samples (same columns as training features)."""
    from joblib import load
    pipe = load(model_path)
    return pipe.predict(samples)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the CSV dataset")
    args = parser.parse_args()

    df = read_data(args.data)
    basic_eda(df)
    results = train_and_compare(df)

    print("\n=== BEST MODEL (by RMSE) ===")
    print(results['best_model'])
    print("\n=== METRICS ===")
    for m, vals in results['metrics'].items():
        print(m, vals)

    print("\nArtifacts saved: metrics_report.csv, best_model.joblib, feature_names.json")

if __name__ == "__main__":
    main()
