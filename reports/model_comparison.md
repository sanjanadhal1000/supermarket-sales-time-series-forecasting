# Model Comparison Report

## Objective
Evaluate and compare ARIMA and LSTM models for time-series forecasting.

---

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

---

## Results

| Model | RMSE   | MAE    | MAPE | Anomalies |
|-------|------  |------  |------|-----------|
| ARIMA | 721.93 | 583.55 | 54.10| 1         |
| LSTM  | 740.07 | 610.45 | 59.29| 0         |

---

## Residual Analysis

Residuals = Actual - Predicted

Residuals were analyzed using Z-score anomaly detection (threshold = 3).

Observations:
- Points beyond 3 standard deviations are flagged as anomalies.
- Lower residual variance indicates better stability.

---

## Conclusion

- ARIMA performs well on structured linear time-series.
- LSTM captures nonlinear patterns but requires more data.
- Based on metrics, select the model with lowest RMSE and MAPE.

For this dataset, the better model is determined from metrics.json.