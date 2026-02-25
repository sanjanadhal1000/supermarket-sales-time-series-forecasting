import numpy as np


# ----------------------------
# Regression Metrics
# ----------------------------

def rmse(actual, predicted):
    """
    Root Mean Squared Error
    Penalizes large errors more heavily.
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    """
    Mean Absolute Error
    Average absolute difference.
    """
    return np.mean(np.abs(actual - predicted))


def mape(actual, predicted):
    """
    Mean Absolute Percentage Error
    Business-friendly percentage metric.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Avoid division by zero
    non_zero = actual != 0
    return np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100


# ----------------------------
# Residual Analysis
# ----------------------------

def residuals(actual, predicted):
    """
    Difference between actual and predicted values.
    """
    return actual - predicted


def detect_anomalies_zscore(residuals, threshold=3):
    """
    Detect anomalies using Z-score method.
    Points beyond threshold standard deviations are anomalies.
    """
    mean = np.mean(residuals)
    std = np.std(residuals)

    z_scores = (residuals - mean) / std
    anomaly_indices = np.where(np.abs(z_scores) > threshold)[0]

    return anomaly_indices, z_scores