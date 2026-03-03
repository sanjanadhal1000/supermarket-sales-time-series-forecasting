📊 Sales Forecasting System

Project Overview

    This project implements an end-to-end sales forecasting system using both traditional statistical modeling (ARIMA) and deep learning (LSTM). The system includes model training, evaluation, comparison, deployment via FastAPI, Dockerization, and basic production monitoring.

    The objective is to build a deployable forecasting pipeline suitable for real-world retail demand planning.

Problem Statement

    - Retail businesses require accurate daily sales forecasts to:
    - Optimize inventory levels
    - Reduce stockouts and overstocking
    - Improve demand planning
    - Detect unusual sales fluctuations early

    This project builds a forecasting system capable of handling time-series sales data and serving predictions through an API.

Dataset Description

    The dataset consists of:

        -Date (Daily frequency)
        -Sales (Aggregated daily sales value)

    Preprocessing steps include:

        -Sorting by date
        -Handling missing dates via interpolation
        -Train-test split (80/20)
        -Time-series validation using walk-forward approach

    No personal or sensitive data is used.

Architecture Diagram

    Raw Sales Data
            ↓
    Preprocessing & Cleaning
            ↓
    Model Training (ARIMA + LSTM)
            ↓
    Evaluation & Comparison
            ↓
    Model Saving
            ↓
    FastAPI Deployment
            ↓
    Docker Containerization
            ↓
    Monitoring (Latency, Drift, Logging)

Model Details

    ARIMA

        - Statistical time-series model
        - Suitable for linear trends and seasonality
        - Lightweight and interpretable

    LSTM (Deep Learning Model)

        -Recurrent Neural Network architecture
        -Captures long-term temporal dependencies
        -Handles non-linear patterns better than ARIMA
        -Performs well during volatile sales periods

    Deep Learning Justification

        LSTM was chosen because:

            - It captures long-term dependencies in time-series data.
            - It models non-linear trends effectively.
            - It performs better during volatile or rapidly changing sales periods.
            - It is suitable for dynamic real-world forecasting scenarios.

Training Process

    - Data split into 80% training and 20% testing
    - ARIMA hyperparameters selected via experimentation
    - LSTM trained using sliding window sequence generation
    - Walk-forward validation applied for robustness
    - Models evaluated using multiple error metrics
    - Best model selected automatically based on RMSE

Evaluation Metrics

The following metrics were used:

    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - Residual analysis
    - Z-score based anomaly detection

Model comparison report is available in:

    reports/model_comparison.md

Performance summary saved in:

reports/performance_summary.csv

Deployment Instructions

    The system is deployed using FastAPI and Docker.

    Build and Run
    docker compose up --build

    API will be available at:

    http://localhost:8000/docs

Available Endpoints

    /health → System health check
    /predict → Single-step prediction
    /forecast → Multi-step forecasting
    /monitor → Monitoring metrics

Monitoring Strategy

    The deployed system includes basic production monitoring:

        - Total API request count
        - Average prediction latency
        - Mean prediction error (drift indicator)
        - Logging using loguru
        - Alert for extreme predictions

    Monitoring data is stored in:

        monitoring/metrics.json

    This ensures visibility into model performance in real-time usage.

Ethical & Edge Case Handling

    - No personal data used → Low privacy risk
    - Missing dates handled via interpolation
    - Alerts generated for extreme predictions
    - No automated decision-making → Human-in-the-loop approach
    - Designed for advisory forecasting, not autonomous execution

Business Impact

    This forecasting system can provide measurable business value:

        - Improved demand planning accuracy
        - Potential reduction in stockouts by 10–15%
        - Early detection of sudden sales drops
        - Better inventory alignment with demand
        - Revenue optimization through proactive planning

    By improving forecast accuracy, retailers can reduce waste, avoid lost sales, and optimize supply chain decisions.

Project Structure

    src/
    api/
    evaluation/
    training/

    reports/
    results/
    monitoring/

    Dockerfile
    docker-compose.yml
    README.md

Conclusion

    This project demonstrates a complete machine learning lifecycle:

        - Data preprocessing
        - Model training
        - Evaluation and comparison
        - Deployment via API
        - Docker containerization
        - Production-style monitoring
        - Business and ethical considerations

    The system is designed to be scalable, modular, and suitable for real-world deployment scenarios.

