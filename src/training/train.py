import pandas as pd
import json                                                          # Allows saving results to JSON File
from src.models.arima_model import grid_search_arima, evaluate_arima # Imports ARIMA functions
from src.models.lstm_model import train_lstm                         # Imports LSTM training function

def main():                                                          # Encapsulates entire pipeline in one function

    # Load processed data
    df = pd.read_csv("data/processed/daily_sales.csv")
    series = df["Total"].values                    # Extracts time-series column, .values convert it to NumPy Array

    # Train-test split (80-20)
    split = int(len(series) * 0.8)                 # Calculates 80% index, seperates training and testing data
    train, test = series[:split], series[split:]   # Splits data chronologically

    # ---------------- ARIMA ----------------
    # Defines hyperparameter search space, controls model complexity
    p_values = [0,1,2]
    d_values = [0,1]
    q_values = [0,1,2]

    arima_model, best_order, best_score = grid_search_arima(          # Finds best ARIMA model
        train, p_values, d_values, q_values
    )

    forecast, arima_rmse, arima_mae = evaluate_arima(arima_model, test) # Evaluates ARIMA on unseen test data

    # ---------------- LSTM ----------------
    lstm_model, lstm_rmse, lstm_mae = train_lstm(train, test)           # Trains LSTM and evaluates it

    # Save metrics
    results = {
        "ARIMA_order": best_order,
        "ARIMA_RMSE": float(arima_rmse), # float types are JSON serializable, whereas NumPy types are not.
        "ARIMA_MAE": float(arima_mae),
        "LSTM_RMSE": float(lstm_rmse),
        "LSTM_MAE": float(lstm_mae)
    }

    with open("results/metrics.json", "w") as f: # Opens file in write mode
        json.dump(results, f, indent=4)          # Writes dictionary into JSON file, indent=4 makes it readable

    # Gives immediate feedback in the terminal
    print("Training complete.") 
    print(results)


if __name__ == "__main__":
    main()
