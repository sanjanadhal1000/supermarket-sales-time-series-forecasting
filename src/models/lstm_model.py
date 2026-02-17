import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
EarlyStopping = tf.keras.callbacks.EarlyStopping


def create_sliding_window(data, window_size):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(train, test, window_size=7):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1,1))
    test_scaled = scaler.transform(test.reshape(-1,1))

    X_train, y_train = create_sliding_window(train_scaled, window_size)
    X_test, y_test = create_sliding_window(test_scaled, window_size)

    model = build_lstm((window_size, 1))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    y_test_actual = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)

    return model, rmse, mae