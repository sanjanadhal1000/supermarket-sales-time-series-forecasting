from fastapi import FastAPI    # Import FASTAPI framework to build REST APIs 
from pydantic import BaseModel # Import BaseModel for request data validation
import numpy as np             # Numerical operations, array handling
import pickle                  # Load the trained ML model
import time                    # Measure API response latency
import json                    # Store monitoring metrics in a JSON file
import os                      # Manage directories and file paths
from loguru import logger      # Structured logging

# Create a FastAPI application instance with a custom API title
app = FastAPI(title="Sales Forecasting API")

# Load ARIMA model
with open("results/arima_model.pkl", "rb") as f: # Open the saved ARIMA model file in binary read mode
    model = pickle.load(f)                       # Load trained ARIMA forecasting model into memory

# Monitoring variables
request_count = 0      # Total no. of prediction reuqests received by the API
total_latency = 0      # Cumulative latency of all API prediction requests
prediction_errors = [] # List to store prediction errors for monitoring model performance

class SalesInput(BaseModel): # Define the input data schema using Pydantic
    values: list[float]      # Expect a list of float values representing historical sales data

# Define a GET endpoint used to check whether the API service is running
@app.get("/health")
def health():
    return {"status": "healthy"} # Return a simple JSON response confirming API health

@app.post("/predict")        # Define POST endpoint to generate single-step sales prediction
def predict(data: SalesInput):
    global request_count, total_latency, prediction_errors  # Access global monitoring values inside the function

    start_time = time.time() # Record the start time to calculate prediction latency

    values = np.array(data.values)        # Convert input sales values into a NumPy array for numerical processing
    forecast = model.forecast(steps=1)[0] # Generate the next predicted sales value using the ARIMA model

    # 🚨 Extreme prediction alert
    if len(values) > 0 and forecast > np.mean(values) * 2: # Check if prediction is abnormally large compared to historical average
        logger.warning("Extreme prediction detected!")     # Log a warning if an unusually high prediction is detected

    latency = time.time() - start_time   # Calculate how long the prediction took to execute
    request_count += 1                   # Increase total API request count
    total_latency += latency             # Add the current latency to cumulative latency for monitoring

    # If actual value provided, compute error
    if len(values) > 0:                  # Ensure there are historical values before computing error
        actual = values[-1]              # Use the last value as the most recent actual sales observation
        error = abs(actual - forecast)   # Calculate absolute prediction error
        prediction_errors.append(error)  # Store error in list to track model performance over time

    logger.info(f"Prediction made. Latency: {latency}") # Log successful prediction along with execution latency

    return {
        "prediction": float(forecast),                  # Return predicted sales value in JSON format
        "latency": latency                              # Return the time taken to generate the prediction
    }

# Define endpoint for generating multi-step sales forecasts
@app.post("/forecast")
def forecast(data: SalesInput, steps: int = 7):         # Accept historical values and number of future steps to forecast
    forecast = model.forecast(steps=steps)              # Generate predictions for the specified number of future periods
    return {"forecast": forecast.tolist()}              # Convert NumPy array to list so it can be returned in JSON format

# Endpoint to retrieve API monitoring metrics
@app.get("/monitor")
def monitor():
    avg_latency = total_latency / request_count if request_count > 0 else 0  # Calculate average prediction latency if requests exist
    mean_error = np.mean(prediction_errors) if prediction_errors else 0      # Calculate mean prediction error if error records exist

    monitoring_data = {
        "request_count": request_count,                  # Total number of prediction requests served by the API
        "average_latency": avg_latency,                  # Average time taken for predictions
        "mean_prediction_error": float(mean_error)       # Average prediction error used as drift indicator
    }

    os.makedirs("monitoring", exist_ok=True)             # Create monitoring directory if it does not already exist
    with open("monitoring/metrics.json", "w") as f:      # Open or create a JSON file to store monitoring data
        json.dump(monitoring_data, f, indent=4)          # Save monitoring metrics to file in readable format

    return monitoring_data                               # Return monitoring statistics as API response

# Root endpoint used as a simple homepage or API health confirmation
@app.get("/")
def home():
    return {"message": "Sales Forecasting API is running"} # Return confirmation message indicating API service is active

# http://localhost:8000/docs