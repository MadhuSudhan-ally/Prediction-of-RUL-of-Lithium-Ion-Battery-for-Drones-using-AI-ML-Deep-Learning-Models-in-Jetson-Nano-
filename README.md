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
