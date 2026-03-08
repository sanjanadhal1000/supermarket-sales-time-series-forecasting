import numpy as np
from src.evaluation.metrics import rmse, mae, mape   # Import evaluation metric functions (RMSE, MAE, MAPE) from the metrics module

# Define a function that performs walk-forward validation on a time-series dataset
# series → the time-series data
# model_func → function used to train the forecasting model
# window_size → optional parameter for rolling window size (not used here but allows extension)
def walk_forward_validation(series, model_func, window_size=30):
    """
    Perform walk-forward validation.
    """

    train_size = int(len(series) * 0.7)                     # Split the dataset so that 70% of the data is used for training
    train, test = series[:train_size], series[train_size:]  # Divide the series into training data and testing data

    history = list(train)                                   # Store training data in a list called history which will be updated step-by-step
    predictions = []                                        # Create an empty list to store predicted values

    for t in range(len(test)):                              # Loop through each observation in the test dataset
        model = model_func(np.array(history))               # Train the model using the current historical data
        yhat = model.forecast()[0]                          # Generate a prediction for the next time step
        predictions.append(yhat)                            # Store the predicted value in the predictions list
        history.append(test[t])                             # Add the actual observed value to history to simulate real-time forecasting

    r = rmse(test, predictions)                             # Root Mean Squared Error between actual values and predictions
    m = mae(test, predictions)                              # Mean Absolute Error between actual values and predictions
    p = mape(test, predictions)                             # Mean Absolute Percentage Error between actual values and predictions

    return r, m, p                                          # Return the calculated evaluation metrics