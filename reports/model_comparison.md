# Advanced Model Evaluation Report

## Methodology

Models evaluated using:

- RMSE
- MAE
- MAPE
- Walk-forward validation
- Residual distribution
- Z-score anomaly detection

---

## Visual Diagnostics

See:
- results/plots/ARIMA_forecast.png
- results/plots/LSTM_forecast.png
- Residual histograms

---

## Model Ranking Criteria

Primary metric: RMSE  
Secondary metric: MAPE  
Tertiary metric: Stability (Residual variance)

---

## Final Decision

Best model is automatically selected and saved in:
results/best_model.txt