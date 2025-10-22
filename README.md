# Prediction-of-RUL-of-Lithium-Ion-Battery-for-Drones-using-AI-ML-Deep-Learning-Models-in-Jetson-Nano-
An AI-based system that predicts the Remaining Useful Life (RUL) of lithium-ion batteries in UAVs using CNN-LSTM deep learning. It processes sensor data to ensure safer, longer flights and is optimized for real-time edge deployment on NVIDIA Jetson Nano.

Overview

This project presents an AI-powered system to predict the Remaining Useful Life (RUL) of lithium-ion batteries used in Unmanned Aerial Vehicles (UAVs). The model uses deep learning (CNN + LSTM) architectures to analyze voltage, current, and temperature data and estimate how long a battery will continue to operate safely.
Optimized for real-time inference on NVIDIA Jetson Nano, this project enhances flight safety, efficiency, and battery maintenance.

🧠 Key Features

Predicts Remaining Useful Life (RUL) of batteries with high accuracy.

Combines Convolutional Neural Networks (CNN) for feature extraction and LSTM for time-series learning.

Provides uncertainty estimates (q10–q90 intervals) for safer decision-making.

Supports real-time deployment on Jetson Nano using TensorFlow Lite or ONNX.

Achieved MAE = 0.45, RMSE = 0.90, with 95% interval coverage.

Enables proactive UAV battery management and extended mission planning.

🧩 Model Architecture

The system processes raw telemetry data (voltage, current, temperature) using:

CNN Layers – Extract local degradation features.

LSTM Layers – Capture temporal evolution of battery health.

Dense Heads – Output RUL mean and confidence intervals (q10, q50, q90).

⚙️ Tech Stack

Programming Language: Python 3.10+

Libraries: TensorFlow / Keras, NumPy, Pandas, Matplotlib

Hardware Used: NVIDIA Jetson Nano

OS: Ubuntu 20.04 (JetPack 4.6+)

Model Format: .keras, .tflite, .onnx



preprocess.py

Create memory-mapped dataset files (X_sequences_memmap.dat, y_labels_memmap.dat)
from a large CSV (LONG or WIDE format). Designed to run in VS Code / Colab.

Usage examples:
  python preprocess.py --csv /path/combined_dataset.csv --outdir /path/output --seq_len 50 --n_features 13
  python preprocess.py --csv /path/combined_dataset.csv --mode wide --seq_len 50 --n_features 13

Behavior:
- WIDE format: each row contains seq_len * n_features flattened features (optionally a y_true column).
- LONG format: data has rows per time-step per unit. Requires an id column and a time column
  (auto-detected); windows of length seq_len are emitted per id once enough rows are available.
- Two-pass approach: first pass counts how many windows will be produced, second pass writes
  memmaps to disk to avoid holding everything in RAM.

Outputs (in --outdir):
 - X_sequences_memmap.dat   (float32) shape=(N_windows, seq_len, n_features)
 - y_labels_memmap.dat      (float32) shape=(N_windows,)
 - metadata.json            (info about seq_len, n_features, N_windows, columns used)

Notes:
- The script is defensive: it tries to auto-detect ID/time columns for LONG format and
  selects numeric feature columns. Non-numeric columns (timestamps, strings) are ignored
  from feature channels.
- For LONG format if id column is missing we create synthetic ids per chunk (not recommended).


