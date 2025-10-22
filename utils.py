

import os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


# GPU setup
def set_gpu():
    """Enable GPU memory growth (optional)."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU ready:", tf.config.list_logical_devices("GPU"))
    except Exception as e:
        print("GPU not configured:", e)


# Data utilities
def load_memmap(path, shape, dtype="float32"):
    """Load .dat file as NumPy memmap."""
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def normalize(X, mean, std, clip=10.0):
    """Normalize data using mean/std."""
    mean3 = mean.reshape(1, 1, -1)
    std3 = np.maximum(std.reshape(1, 1, -1), 1e-6)
    X = np.where(np.isfinite(X), X, mean3)
    X = (X - mean3) / std3
    return np.clip(X, -clip, clip).astype("float32")


# Model helpers
def build_cnn_lstm(seq_len=50, n_feat=13):
    """Build CNN + LSTM model."""
    inp = tf.keras.Input(shape=(seq_len, n_feat))
    x = tf.keras.layers.Conv1D(96, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    out_mean = tf.keras.layers.Dense(1, name="mean")(x)
    out_q10 = tf.keras.layers.Dense(1, name="q10")(x)
    out_q50 = tf.keras.layers.Dense(1, name="q50")(x)
    out_q90 = tf.keras.layers.Dense(1, name="q90")(x)
    return tf.keras.Model(inp, [out_mean, out_q10, out_q50, out_q90])


# Metrics and saving
def compute_metrics(y_true, y_pred, q10=None, q90=None):
    e = y_pred - y_true
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    metrics = {"MAE": mae, "RMSE": rmse}
    if q10 is not None and q90 is not None:
        coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
        metrics["Coverage"] = coverage
    return metrics


def save_json(data, path):
    """Save dict as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def plot_results(pred_mean, q10, q90, y_true):
    """Quick result visualization."""
    plt.figure(); plt.hist(pred_mean, bins=40)
    plt.title("Predicted RUL"); plt.xlabel("RUL"); plt.show()
    plt.figure(); plt.scatter(pred_mean, q90 - q10, s=5)
    plt.title("Interval Width vs Mean"); plt.show()
    plt.figure(); plt.hist(pred_mean - y_true, bins=40)
    plt.title("Residuals"); plt.show()


# Export utilities
def export_tflite(model, out_path="model.tflite", quantize=False):
    """Convert model to TFLite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {out_path}")
