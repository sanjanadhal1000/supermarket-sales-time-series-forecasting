import pandas as pd
import numpy as np
import json
import pickle
import os

from src.models.arima_model import grid_search_arima, evaluate_arima  # Import ARIMA training and evaluation functions
from src.models.lstm_model import train_lstm                          # Import function to train the LSTM deep learning model
from src.evaluation.metrics import rmse, mae, mape, residuals, detect_anomalies_zscore # Import evaluation metrics and anomaly detection functions

# Main function that executes the entire training and evaluation pipeline
def main():

    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)                             # Create a results directory if it does not already exist

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv("data/processed/daily_sales.csv")
    series = df["Total"].values                          # Extract the sales column as a NumPy array for modeling

    # Train-test split (80-20)
    split = int(len(series) * 0.8)                       # Calculate the split index for 80% training data
    train, test = series[:split], series[split:]         # Split the dataset into training and testing sets

    # ==========================================================
    #                       ARIMA
    # ==========================================================
    p_values = [0, 1, 2]                                 # Define possible autoregressive orders for ARIMA
    d_values = [0, 1]                                    # Define possible differencing values
    q_values = [0, 1, 2]                                 # Define possible moving average orders

    arima_model, best_order, _ = grid_search_arima(
        train, p_values, d_values, q_values
    )                                                    # Perform grid search to find the best ARIMA parameters

    forecast, _, _ = evaluate_arima(arima_model, test)   # Generate ARIMA predictions on test data

    # ARIMA Metrics
    arima_rmse = rmse(test, forecast)                    # Root Mean Squared Error
    arima_mae = mae(test, forecast)                      # Mean Absolute Error
    arima_mape = mape(test, forecast)                    # Mean Absolute Percentage Error

    # ARIMA Residuals
    arima_residuals = residuals(test, forecast)          # Residual errors (actual − predicted)
    arima_anomalies, _ = detect_anomalies_zscore(arima_residuals)  # Detect anomalies in residuals using Z-score method

    # Save ARIMA model
    with open("results/arima_model.pkl", "wb") as f:
        pickle.dump(arima_model, f)                      # Save the trained ARIMA model to disk

    # ==========================================================
    #                       LSTM
    # ==========================================================
    lstm_model, lstm_rmse, lstm_mae = train_lstm(train, test) # Train the LSTM model and obtain basic evaluation metrics

    # Recalculate predictions for MAPE & residuals
    from src.models.lstm_model import create_sliding_window   # Import function to create time-series input sequences
    from sklearn.preprocessing import MinMaxScaler            # Import scaler for normalizing data

    scaler = MinMaxScaler()                                   # Initialize MinMaxScaler for feature scaling
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)) # Scale training data to range [0,1]
    test_scaled = scaler.transform(test.reshape(-1, 1))       # Apply same scaling to test data

    window_size = 7                                           # Define time window size for LSTM sequences
    X_test, y_test = create_sliding_window(test_scaled, window_size) # Create input-output sequences for LSTM testing

    predictions = lstm_model.predict(X_test)                  # Generate predictions using trained LSTM model
    predictions = scaler.inverse_transform(predictions)       # Convert scaled predictions back to original scale
    y_test_actual = scaler.inverse_transform(y_test)          # Convert scaled actual values back to original scale

    lstm_mape = mape(y_test_actual, predictions)              # Calculate MAPE for LSTM predictions

    lstm_residuals = residuals(y_test_actual.flatten(), predictions.flatten())   # Compute residual errors for LSTM
    lstm_anomalies, _ = detect_anomalies_zscore(lstm_residuals)                  # Detect anomalies in LSTM residuals

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

    # Compare based on RMSE
    if arima_rmse < lstm_rmse:
        best_model = "ARIMA"
    else:
        best_model = "LSTM"

    with open("results/best_model.txt", "w") as f:
        f.write(f"Best model based on RMSE: {best_model}")

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Training & Evaluation Complete")
    print(json.dumps(results, indent=4))



    summary_df = pd.DataFrame(results).T
    summary_df.to_csv("reports/performance_summary.csv")


if __name__ == "__main__":
    main()
