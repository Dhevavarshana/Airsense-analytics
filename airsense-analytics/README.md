# AirSense Analytics: AI-Powered Aviation Intelligence & Decision Support Platform

An end-to-end machine learning platform for aviation intelligence, flight delay forecasting, and cancellation risk evaluation.

## Project Structure

```text
airsense-analytics/
├── data/
│   ├── generate_synthetic_data.py   # Realistic 50k synthetic flight dataset generator
│   └── flights.csv                  # Generated dataset (50,000 rows x 17 columns)
├── src/
│   ├── train_models.py              # Model training pipeline (Classifier, Regressor, Cancellation)
│   └── predict_sample.py            # Automated verification & sample inference test
├── models/
│   ├── best_delay_classifier.joblib # Saved best delay classifier (XGBoost)
│   ├── best_delay_regressor.joblib  # Saved best delay duration regressor (XGBoost)
│   ├── best_cancellation_classifier.joblib # Saved best cancellation classifier (Random Forest)
│   └── metrics.json                 # Comprehensive benchmark metrics
├── metrics.json                     # Benchmark metrics (root mirror)
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Documentation
```

## Features & Correlation Design
The synthetic dataset generator (`data/generate_synthetic_data.py`) models realistic aviation dynamics:
- **Weather Impact**: Severe weather conditions (Thunderstorm, Snow, Fog, Rain) scale delay minutes, wind speed variance, visibility limits, and cancellation likelihoods.
- **Hour & Congestion**: Peak evening hours (16:00 - 21:00) simulate compounding delays across hubs compared to early morning departures.
- **Airline Baselines**: Carrier-specific delay offsets reflecting operational efficiency differences.

## Execution

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate synthetic dataset**:
   ```bash
   python data/generate_synthetic_data.py
   ```

3. **Train and benchmark models**:
   ```bash
   python src/train_models.py
   ```

4. **Verify all models and test sample predictions**:
   ```bash
   python src/predict_sample.py
   ```

5. **Start the Live Web Platform**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to: **`http://localhost:8050`**
