import pandas as pd
import numpy as np
import json
import pickle
import os

from src.models.arima_model import grid_search_arima, evaluate_arima
from src.models.lstm_model import train_lstm
from src.evaluation.metrics import rmse, mae, mape, residuals, detect_anomalies_zscore


def main():

    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv("data/processed/daily_sales.csv")
    series = df["Total"].values

    # Train-test split (80-20)
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    # ==========================================================
    #                       ARIMA
    # ==========================================================
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    arima_model, best_order, _ = grid_search_arima(
        train, p_values, d_values, q_values
    )

    forecast, _, _ = evaluate_arima(arima_model, test)

    # ARIMA Metrics
    arima_rmse = rmse(test, forecast)
    arima_mae = mae(test, forecast)
    arima_mape = mape(test, forecast)

    # ARIMA Residuals
    arima_residuals = residuals(test, forecast)
    arima_anomalies, _ = detect_anomalies_zscore(arima_residuals)

    # Save ARIMA model
    with open("results/arima_model.pkl", "wb") as f:
        pickle.dump(arima_model, f)

    # ==========================================================
    #                       LSTM
    # ==========================================================
    lstm_model, lstm_rmse, lstm_mae = train_lstm(train, test)

    # Recalculate predictions for MAPE & residuals
    from src.models.lstm_model import create_sliding_window
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))

    window_size = 7
    X_test, y_test = create_sliding_window(test_scaled, window_size)

    predictions = lstm_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)

    lstm_mape = mape(y_test_actual, predictions)

    lstm_residuals = residuals(y_test_actual.flatten(), predictions.flatten())
    lstm_anomalies, _ = detect_anomalies_zscore(lstm_residuals)

    # Save LSTM model
    lstm_model.save("results/lstm_model.h5")

    # ==========================================================
    # Save Metrics
    # ==========================================================

    results = {
        "ARIMA": {
            "order": best_order,
            "RMSE": float(arima_rmse),
            "MAE": float(arima_mae),
            "MAPE": float(arima_mape),
            "Anomalies_detected": int(len(arima_anomalies))
        },
        "LSTM": {
            "RMSE": float(lstm_rmse),
            "MAE": float(lstm_mae),
            "MAPE": float(lstm_mape),
            "Anomalies_detected": int(len(lstm_anomalies))
        }
    }

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Training & Evaluation Complete")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
