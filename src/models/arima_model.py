import itertools                               # Provides tools to work with combinations
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # ARIMA model to be used for forecasting
from sklearn.metrics import mean_squared_error, mean_absolute_error

def grid_search_arima(train, p_values, d_values, q_values): # Finds the best ARIMA parameters
    best_score = float("inf")                               # Sets initial error to infinity
    best_order = None                                       # Store best (p,d,q)
    best_model = None                                       # Store the best trained model

    for order in itertools.product(p_values, d_values, q_values): # Grid search (loop through all combinations of p,d,q)
        try:                                                      # Prevents crash if ARIMA fails
            model = ARIMA(train, order=order)                     # Creates ARIMA model
            model_fit = model.fit()                               # Trains it on training data

            predictions = model_fit.fittedvalues                  # Predicted values for training data
            rmse = np.sqrt(mean_squared_error(train, predictions)) # Calculates model error

            if rmse < best_score:                                 # Checks if the model is better
                best_score = rmse
                best_order = order
                best_model = model_fit
        except:
            continue

    return best_model, best_order, best_score                    # Best trained model, best (p,d,q), best RMSE

def evaluate_arima(model, test):                                 # Evaluates trained model on unseen test data
    forecast = model.forecast(steps=len(test))                   # Predicts future values
    rmse = np.sqrt(mean_squared_error(test, forecast))           # Penalizes large errors
    mae = mean_absolute_error(test, forecast)                    # Average Error Magnitude

    return forecast, rmse, mae