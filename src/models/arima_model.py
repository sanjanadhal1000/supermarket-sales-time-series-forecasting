import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

def grid_search_arima(train, p_values, d_values, q_values):
    best_score = float("inf")
    best_order = None
    best_model = None

    for order in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()

            predictions = model_fit.fittedvalues
            rmse = np.sqrt(mean_squared_error(train, predictions))

            if rmse < best_score:
                best_score = rmse
                best_order = order
                best_model = model_fit
        except:
            continue

    return best_model, best_order, best_score

def evaluate_arima(model, test):
    forecast = model.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    return forecast, rmse, mae