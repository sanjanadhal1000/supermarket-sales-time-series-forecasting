import matplotlib.pyplot as plt
import os


def plot_forecast(actual, predicted, model_name):
    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(f"{model_name} Forecast vs Actual")
    plt.legend()
    plt.savefig(f"results/plots/{model_name}_forecast.png")
    plt.close()


def plot_residuals(residuals, model_name):
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30)
    plt.title(f"{model_name} Residual Distribution")
    plt.savefig(f"results/plots/{model_name}_residuals.png")
    plt.close()