import numpy as np
from src.evaluation.metrics import rmse, mae, mape


def walk_forward_validation(series, model_func, window_size=30):
    """
    Perform walk-forward validation.
    """

    train_size = int(len(series) * 0.7)
    train, test = series[:train_size], series[train_size:]

    history = list(train)
    predictions = []

    for t in range(len(test)):
        model = model_func(np.array(history))
        yhat = model.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    r = rmse(test, predictions)
    m = mae(test, predictions)
    p = mape(test, predictions)

    return r, m, p