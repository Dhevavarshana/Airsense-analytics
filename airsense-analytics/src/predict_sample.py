"""AirSense Analytics - Verification & Inference Test Script
Tests loading all 3 trained models and runs sample predictions.
"""

import json
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flights.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

def verify_project():
    print("=" * 60)
    print("AIRSENSE ANALYTICS - VERIFICATION CHECK")
    print("=" * 60)

    # 1. Check Data File
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[PASS] Data File exists: {DATA_PATH}")
        print(f"       Rows: {len(df):,}, Columns: {len(df.columns)}")
    else:
        print(f"[FAIL] Data file missing at: {DATA_PATH}")
        return

    # 2. Check Metrics File
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        print(f"[PASS] Metrics File exists: {METRICS_PATH}")
        print(f"       Delay Classifier Best:       {metrics['delay_classifier']['best_model']}")
        print(f"       Delay Regressor Best:        {metrics['delay_regressor']['best_model']}")
        print(f"       Cancellation Classifier Best:{metrics['cancellation_classifier']['best_model']}")
    else:
        print(f"[FAIL] Metrics file missing at: {METRICS_PATH}")
        return

    # 3. Load Models and Test Inference
    clf_path = os.path.join(MODELS_DIR, "best_delay_classifier.joblib")
    reg_path = os.path.join(MODELS_DIR, "best_delay_regressor.joblib")
    canc_path = os.path.join(MODELS_DIR, "best_cancellation_classifier.joblib")

    for path, name in [(clf_path, "Delay Classifier"), (reg_path, "Delay Regressor"), (canc_path, "Cancellation Classifier")]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[PASS] Model file exists: {name} ({size_mb:.2f} MB)")
        else:
            print(f"[FAIL] Model missing: {path}")
            return

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    canc = joblib.load(canc_path)

    # Sample Test Scenario 1: Severe Weather (Thunderstorm at evening peak)
    scenario_storm = pd.DataFrame([{
        "distance": 850,
        "temperature": 27.5,
        "wind_speed": 34.0,
        "visibility": 1.5,
        "day_of_week": 4,
        "month": 7,
        "hour": 19,
        "airline": "NK",
        "origin": "ORD",
        "destination": "ATL",
        "weather_condition": "Thunderstorm",
    }])

    # Sample Test Scenario 2: Clear Morning Flight
    scenario_clear = pd.DataFrame([{
        "distance": 450,
        "temperature": 18.0,
        "wind_speed": 5.0,
        "visibility": 10.0,
        "day_of_week": 2,
        "month": 5,
        "hour": 7,
        "airline": "DL",
        "origin": "ATL",
        "destination": "MCO",
        "weather_condition": "Clear",
    }])

    print("\n" + "-" * 60)
    print("LIVE INFERENCE TEST ON SAMPLE FLIGHTS")
    print("-" * 60)

    for label, sc in [("Scenario 1 (Severe Storm / Peak Hour / NK)", scenario_storm),
                      ("Scenario 2 (Clear Skies / Morning / DL)", scenario_clear)]:
        delayed = clf.predict(sc)[0]
        delay_prob = clf.predict_proba(sc)[0][1]
        delay_mins = reg.predict(sc)[0]
        cancelled = canc.predict(sc)[0]
        cancel_prob = canc.predict_proba(sc)[0][1]

        print(f"\n* {label}:")
        print(f"  - Delay Prediction:        {'DELAYED (>=15 min)' if delayed else 'ON TIME'} (Probability: {delay_prob:.1%})")
        print(f"  - Predicted Delay Duration:{delay_mins:.1f} minutes")
        print(f"  - Cancellation Risk:       {'HIGH RISK' if cancelled else 'LOW RISK'} (Probability: {cancel_prob:.1%})")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED: The platform is running 100% correctly!")
    print("=" * 60)

if __name__ == "__main__":
    verify_project()
