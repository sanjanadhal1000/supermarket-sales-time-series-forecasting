import pandas as pd
import numpy as np
import json
from src.models.arima_model import grid_search_arima, evaluate_arima
from src.models.lstm_model import train_lstm

def main():

    # Load processed data
    df = pd.read_csv("data/processed/daily_sales.csv")
    series = df["Total"].values

    # Train-test split (80-20)
    split = int(len(series) * 0.8)
    train, test = series[:split], series[split:]

    # ---------------- ARIMA ----------------
    p_values = [0,1,2]
    d_values = [0,1]
    q_values = [0,1,2]

    arima_model, best_order, best_score = grid_search_arima(
        train, p_values, d_values, q_values
    )

    forecast, arima_rmse, arima_mae = evaluate_arima(arima_model, test)

    # ---------------- LSTM ----------------
    lstm_model, lstm_rmse, lstm_mae = train_lstm(train, test)

    # Save metrics
    results = {
        "ARIMA_order": best_order,
        "ARIMA_RMSE": float(arima_rmse),
        "ARIMA_MAE": float(arima_mae),
        "LSTM_RMSE": float(lstm_rmse),
        "LSTM_MAE": float(lstm_mae)
    }

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Training complete.")
    print(results)


if __name__ == "__main__":
    main()
