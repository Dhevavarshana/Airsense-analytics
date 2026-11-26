"""AirSense Analytics - Model Training Pipeline
Trains delay classifier, delay regressor, and cancellation classifier
using Logistic Regression, Random Forest, and XGBoost.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flights.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature definitions
NUMERIC_FEATURES = [
    "distance", "temperature", "wind_speed", "visibility",
    "day_of_week", "month", "hour"
]
CATEGORICAL_FEATURES = ["airline", "origin", "destination", "weather_condition"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Creates standard numerical scaler and categorical one-hot encoder."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Compute classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC."""
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probas)), 4),
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    """Compute regression metrics: RMSE, MAE, R2."""
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


def train_delay_classifier(df: pd.DataFrame) -> tuple:
    """Task A: Classify if a flight will be delayed (arrival_delay >= 15 min)."""
    print("\n--- Training Delay Classifier (Task A) ---")
    active_df = df[df["cancelled"] == 0].copy()
    active_df["is_delayed"] = (active_df["arrival_delay"] >= 15).astype(int)

    X = active_df[ALL_FEATURES]
    y = active_df["is_delayed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=90, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=-1),
    }

    metrics = {}
    best_name, best_f1, best_pipe = None, -1.0, None

    for name, clf in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        score = evaluate_classifier(pipe, X_test, y_test)
        metrics[name] = score
        print(f"[{name}] {score}")
        if score["f1"] > best_f1:
            best_f1 = score["f1"]
            best_name = name
            best_pipe = pipe

    return best_name, best_pipe, metrics


def train_delay_regressor(df: pd.DataFrame) -> tuple:
    """Task B: Predict delay duration in minutes for operating flights."""
    print("\n--- Training Delay Regressor (Task B) ---")
    active_df = df[df["cancelled"] == 0].copy()
    X = active_df[ALL_FEATURES]
    y = active_df["arrival_delay"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=90, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
    }

    metrics = {}
    best_name, best_rmse, best_pipe = None, float("inf"), None

    for name, reg in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("regressor", reg)])
        pipe.fit(X_train, y_train)
        score = evaluate_regressor(pipe, X_test, y_test)
        metrics[name] = score
        print(f"[{name}] {score}")
        if score["rmse"] < best_rmse:
            best_rmse = score["rmse"]
            best_name = name
            best_pipe = pipe

    return best_name, best_pipe, metrics


def train_cancellation_classifier(df: pd.DataFrame) -> tuple:
    """Task C: Predict flight cancellation risk."""
    print("\n--- Training Cancellation Classifier (Task C) ---")
    X = df[ALL_FEATURES]
    y = df["cancelled"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Class balance adjustment for cancellations
    neg, pos = np.bincount(y_train)
    pos_weight = neg / max(pos, 1)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=90, max_depth=6, scale_pos_weight=pos_weight, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=-1),
    }

    metrics = {}
    best_name, best_f1, best_pipe = None, -1.0, None

    for name, clf in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        score = evaluate_classifier(pipe, X_test, y_test)
        metrics[name] = score
        print(f"[{name}] {score}")
        if score["f1"] > best_f1:
            best_f1 = score["f1"]
            best_name = name
            best_pipe = pipe

    return best_name, best_pipe, metrics


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Flight dataset not found at {DATA_PATH}. Run generate_synthetic_data.py first.")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    all_metrics = {}

    # 1. Delay Classifier
    best_clf_name, best_clf_pipe, clf_metrics = train_delay_classifier(df)
    clf_model_path = os.path.join(MODELS_DIR, "best_delay_classifier.joblib")
    joblib.dump(best_clf_pipe, clf_model_path)
    all_metrics["delay_classifier"] = {
        "best_model": best_clf_name,
        "saved_path": clf_model_path,
        "comparison": clf_metrics,
    }

    # 2. Delay Regressor
    best_reg_name, best_reg_pipe, reg_metrics = train_delay_regressor(df)
    reg_model_path = os.path.join(MODELS_DIR, "best_delay_regressor.joblib")
    joblib.dump(best_reg_pipe, reg_model_path)
    all_metrics["delay_regressor"] = {
        "best_model": best_reg_name,
        "saved_path": reg_model_path,
        "comparison": reg_metrics,
    }

    # 3. Cancellation Classifier
    best_canc_name, best_canc_pipe, canc_metrics = train_cancellation_classifier(df)
    canc_model_path = os.path.join(MODELS_DIR, "best_cancellation_classifier.joblib")
    joblib.dump(best_canc_pipe, canc_model_path)
    all_metrics["cancellation_classifier"] = {
        "best_model": best_canc_name,
        "saved_path": canc_model_path,
        "comparison": canc_metrics,
    }

    # Save metrics.json in models directory and project root
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    root_metrics_path = os.path.join(BASE_DIR, "metrics.json")
    with open(root_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print("\n================ Training Complete ================")
    print(f"Best Delay Classifier:       {best_clf_name} -> {clf_model_path}")
    print(f"Best Delay Regressor:        {best_reg_name} -> {reg_model_path}")
    print(f"Best Cancellation Classifier:{best_canc_name} -> {canc_model_path}")
    print(f"Metrics saved to:            {metrics_path}")


if __name__ == "__main__":
    main()
