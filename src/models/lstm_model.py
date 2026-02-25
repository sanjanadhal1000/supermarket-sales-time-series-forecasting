import numpy as np
from sklearn.preprocessing import MinMaxScaler                      # Scales values between 0 and 1
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf                                             # Deep learning framework

# Shortcuts to avoid writing tf.keras... repeatedly
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
EarlyStopping = tf.keras.callbacks.EarlyStopping


def create_sliding_window(data, window_size):           # Transforms time-series into supervised learning format
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])                 # X - past window
        y.append(data[i+window_size])                   # y - next value

    return np.array(X), np.array(y)                     # Neural Networks require NumPy arrays

def build_lstm(input_shape):                            # Encapsulates model creation
    model = Sequential()                                # Stack layers in order
    model.add(LSTM(50, activation='relu', input_shape=input_shape)) # Adds LSTM layer with 50 neurons and ReLU activation
    model.add(Dense(1))                                 # Outputs 1 predicted value
    model.compile(optimizer='adam', loss='mse')         # Updates weight, measures error
    return model


def train_lstm(train, test, window_size=7):             # Full training pipeline
    scaler = MinMaxScaler()                             # Creates scaler object
    train_scaled = scaler.fit_transform(train.reshape(-1,1)) # Reshape as scaler expects 2D array
    test_scaled = scaler.transform(test.reshape(-1,1))       # transform used to avoid data leakage

    X_train, y_train = create_sliding_window(train_scaled, window_size) # Transforms scaled data into supervised format
    X_test, y_test = create_sliding_window(test_scaled, window_size)

    model = build_lstm((window_size, 1))                     # Architecture

    early_stop = EarlyStopping(                              # Prevents overfitting
        monitor='val_loss',
        patience=10,                     # Stop training if validation loss doesn't improve for 10 epochs
        restore_best_weights=True
    )

    model.fit(                           # Trains model
        X_train, y_train,
        validation_data=(X_test, y_test), # Monitors performance
        epochs=100,                       # Max iterations
        batch_size=16,                    # Samples per update
        callbacks=[early_stop],
        verbose=1
    )

    predictions = model.predict(X_test)   # Generates forecasts
    predictions = scaler.inverse_transform(predictions) # Model predicts scaled values (0-1). Convert back to original price scale

    y_test_actual = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)

    return model, rmse, mae               # Trained model, evaluation results