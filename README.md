# AIS Ship Anomaly Detection with RNNs and LSTMs

This project uses AIS vessel tracking data to detect unusual ship movement patterns. The main idea is to train sequence models on recent vessel movement history and use prediction error as an anomaly score.

Instead of directly predicting the next absolute latitude and longitude, the model predicts the vessel’s movement delta:

```text
target_delta = true_next_position - last_observed_position
```

The predicted delta is then added back to the last observed position:

```text
predicted_next_position = last_observed_position + predicted_delta
```

This makes the task more realistic because ships usually move locally, and predicting movement is easier than predicting global coordinates.

## Project Goal

Given a sequence of past AIS points for a vessel, predict where the vessel should go next. If the predicted next point is far from the real next point, that high prediction error may indicate anomalous movement.

## Dataset

This project uses AIS ship tracking data from Kaggle.

Dataset used:

```text
AIS Ship Tracking: Vessel Dynamics and ETA Data
```

The dataset includes vessel movement information such as:

- MMSI vessel ID
- Timestamp
- Latitude
- Longitude
- Speed over ground
- Course over ground
- Heading

## Features Used

Each timestep contains six features:

```text
lat, lon, sog, cog, heading, time_gap_minutes
```

Each model input is a sequence of 20 AIS observations:

```text
[20 timesteps, 6 features]
```

The model target is the next movement delta:

```text
[delta_lat, delta_lon]
```

## Models

This project compares three approaches:

### 1. Persistence Baseline

The persistence baseline predicts that the next position is simply the last observed position.

```text
predicted_next_position = last_observed_position
```

This is a strong baseline because ships often move only a small distance between AIS updates.

### 2. Vanilla RNN

The vanilla RNN uses a standard recurrent neural network to learn vessel movement patterns from past AIS sequences.

### 3. Custom Gate LSTM

The LSTM model uses a manually implemented LSTM-style recurrent cell. It exposes the input, forget, and output gates so the model’s gate behavior can be inspected around normal and anomalous examples.

## Evaluation

The project uses Haversine distance to measure prediction error in kilometers.

```text
error_km = Haversine(predicted_next_position, true_next_position)
```

This is better than plain MSE because latitude and longitude represent real positions on Earth.

## Anomaly Detection

After training, the LSTM prediction error is used as an anomaly score.

The project computes:

- Global anomaly thresholds
- Per-region anomaly thresholds
- Per-ship anomaly thresholds

A point is flagged as anomalous if its prediction error is unusually high compared to validation errors.

## Visualizations

The notebook includes:

- Training and validation loss curves
- Baseline vs RNN vs LSTM error comparison
- Error distribution histograms
- Anomaly maps
- Prediction error intensity maps
- Top anomaly path visualization
- LSTM gate behavior plots for anomalous and normal examples

## Key Idea

A normal vessel should move in a way that is predictable from its recent path. If the model expects the ship to move one way, but the ship appears somewhere very different, the large prediction error can indicate unusual behavior.

## Tech Stack

- Python
- Pandas
- NumPy
- PyTorch
- scikit-learn
- Matplotlib
- Kaggle Notebooks

## Status

This project is an experimental machine learning pipeline for AIS-based vessel anomaly detection. It focuses on sequence modeling, movement prediction, real-world geospatial error measurement, and anomaly scoring.
