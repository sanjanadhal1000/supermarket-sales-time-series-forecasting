from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import time
import json
import os
from loguru import logger

app = FastAPI(title="Sales Forecasting API")

# Load ARIMA model
with open("results/arima_model.pkl", "rb") as f:
    model = pickle.load(f)

# Monitoring variables
request_count = 0
total_latency = 0
prediction_errors = []

class SalesInput(BaseModel):
    values: list[float]

@app.get("/health")
def health():
    return {"status": "healthy"}

prediction_errors = []

@app.post("/predict")
def predict(data: SalesInput):
    global request_count, total_latency, prediction_errors

    start_time = time.time()

    values = np.array(data.values)
    forecast = model.forecast(steps=1)[0]

    # 🚨 Extreme prediction alert
    if len(values) > 0 and forecast > np.mean(values) * 2:
        logger.warning("Extreme prediction detected!")

    latency = time.time() - start_time
    request_count += 1
    total_latency += latency

    # If actual value provided, compute error
    if len(values) > 0:
        actual = values[-1]
        error = abs(actual - forecast)
        prediction_errors.append(error)

    logger.info(f"Prediction made. Latency: {latency}")

    return {
        "prediction": float(forecast),
        "latency": latency
    }

@app.post("/forecast")
def forecast(data: SalesInput, steps: int = 7):
    forecast = model.forecast(steps=steps)
    return {"forecast": forecast.tolist()}

@app.get("/monitor")
def monitor():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    mean_error = np.mean(prediction_errors) if prediction_errors else 0

    monitoring_data = {
        "request_count": request_count,
        "average_latency": avg_latency,
        "mean_prediction_error": float(mean_error)
    }

    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/metrics.json", "w") as f:
        json.dump(monitoring_data, f, indent=4)

    return monitoring_data

@app.get("/")
def home():
    return {"message": "Sales Forecasting API is running"}

# http://localhost:8000/docs