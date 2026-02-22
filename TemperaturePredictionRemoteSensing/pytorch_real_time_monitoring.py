# pytorch_real_time_monitoring.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from K_LSTM_Model import KLSTM
from BiLSTM_Attention_Model import BiLSTMAttention
from Convolutional_Autoencoder import ConvAutoencoder


class ColdChainMonitor:
    """
    Real-time monitoring system for cold chain temperature prediction using PyTorch.
    Loads a trained model and preprocessor, and generates alerts based on predictions.
    """

    def __init__(self, model_path, model_class, preprocessor_path, input_dim, seq_len,
                 prediction_horizon=6, threshold_c=6.0, device=None):
        """
        Args:
            model_path: Path to saved PyTorch model state dict (.pth)
            model_class: The model class (KLSTM or BiLSTMAttention)
            preprocessor_path: Path to saved preprocessor object (pickle)
            input_dim: Number of input features (must match model)
            seq_len: Sequence length used during training
            prediction_horizon: Number of future timesteps to predict
            threshold_c: Temperature threshold for alerts (°C)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        # Load model state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Initialize model
        self.model = model_class(
            input_dim=input_dim,
            seq_len=seq_len,
            prediction_horizon=prediction_horizon
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.threshold = threshold_c
        self.alert_history = []

        # Buffer to store recent readings for sequence construction
        self.buffer = []  # list of (timestamp, feature_vector)

        print("ColdChainMonitor called InIt...")

    def preprocess_current_data(self, current_sensor_data, remote_sensing_data):
        """
        Combine current sensor and remote sensing data into a feature vector,
        then scale and prepare for model input.
        For simplicity, this function assumes you will maintain the sequence buffer separately.
        In production, you would append this vector to a rolling buffer.
        """
        # Combine into a single dict
        combined = {**current_sensor_data, **remote_sensing_data}

        # Convert to DataFrame row
        df = pd.DataFrame([combined])

        # Apply time feature creation (if your preprocessor expects them)
        if hasattr(self.preprocessor, 'create_time_features'):
            df = self.preprocessor.create_time_features(df)

        # Ensure all required features exist and are in correct order
        required_features = self.preprocessor.feature_columns  # stored during training
        for feat in required_features:
            if feat not in df.columns:
                df[feat] = 0.0  # fallback for missing features

        # Scale the features using the preprocessor's scaler
        X_scaled = self.preprocessor.feature_scaler.transform(df[required_features].values)

        return X_scaled[0]  # return 1D array of features

    def update_buffer(self, feature_vector, timestamp):
        """Add a new feature vector to the rolling buffer."""
        self.buffer.append((timestamp, feature_vector))
        # Keep only the most recent seq_len entries
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def get_current_sequence(self):
        """
        Build a sequence from the buffer. If buffer length < seq_len,
        pad with the earliest available vector (repeat edge).
        Returns a numpy array of shape (1, seq_len, n_features)
        """
        if len(self.buffer) == 0:
            raise RuntimeError("Buffer is empty. Call update_buffer first.")

        n_features = len(self.buffer[0][1])
        sequence = []

        if len(self.buffer) >= self.seq_len:
            # Use the last seq_len entries
            for i in range(-self.seq_len, 0):
                sequence.append(self.buffer[i][1])
        else:
            # Pad by repeating the first element
            first_vec = self.buffer[0][1]
            pad_count = self.seq_len - len(self.buffer)
            for _ in range(pad_count):
                sequence.append(first_vec)
            for _, vec in self.buffer:
                sequence.append(vec)

        sequence = np.array(sequence)  # (seq_len, n_features)
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)  # (1, seq_len, n_features)
        return sequence

    def predict(self):
        """
        Make a prediction using the current buffer.
        Returns predicted temperatures (numpy array of shape (prediction_horizon,)).
        """
        if len(self.buffer) < 1:
            raise RuntimeError("Insufficient data in buffer for prediction.")

        # Get sequence
        X_seq = self.get_current_sequence()
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()  # (1, pred_horizon)

        # Inverse transform predictions to original scale
        # The preprocessor's target_scaler expects 2D array (n_samples, 1)
        y_pred_flat = y_pred_scaled.flatten().reshape(-1, 1)
        y_pred_orig = self.preprocessor.target_scaler.inverse_transform(y_pred_flat).flatten()

        return y_pred_orig

    def check_alerts(self, predictions):
        """
        Evaluate predictions against thresholds and generate alerts.
        Returns list of alert dicts.
        """
        alerts = []
        max_pred = np.max(predictions)
        min_pred = np.min(predictions)

        # High temperature alert
        if max_pred > self.threshold:
            alerts.append({
                'timestamp': datetime.now(),
                'type': 'HIGH_TEMP_RISK',
                'message': f'Predicted temperature exceeds threshold: {max_pred:.1f}°C > {self.threshold}°C',
                'severity': 'CRITICAL' if max_pred > self.threshold + 2 else 'WARNING',
                'max_predicted': float(max_pred),
                'predictions': predictions.tolist()
            })

        # Rapid rise alert (increase >1°C between consecutive predicted timesteps)
        diffs = np.diff(predictions)
        if np.any(diffs > 1.0):
            alerts.append({
                'timestamp': datetime.now(),
                'type': 'RAPID_TEMP_RISE',
                'message': f'Rapid temperature increase detected (max rise: {np.max(diffs):.2f}°C)',
                'severity': 'WARNING',
                'max_rise': float(np.max(diffs)),
                'predictions': predictions.tolist()
            })

        # Store alerts in history
        self.alert_history.extend(alerts)
        return alerts

    def step(self, current_sensor_data, remote_sensing_data):
        """
        Process one new observation: preprocess, update buffer, predict, and check alerts.
        Returns dict with predictions and alerts.
        """
        # Preprocess new data point
        feature_vec = self.preprocess_current_data(current_sensor_data, remote_sensing_data)

        # Update buffer with timestamp and feature vector
        self.update_buffer(feature_vec, current_sensor_data.get('timestamp', datetime.now()))

        # If buffer is full enough, make prediction
        if len(self.buffer) >= self.seq_len:
            predictions = self.predict()
            alerts = self.check_alerts(predictions)
        else:
            # Not enough history yet
            predictions = np.full(self.prediction_horizon, np.nan)
            alerts = [{
                'timestamp': datetime.now(),
                'type': 'INSUFFICIENT_DATA',
                'message': f'Buffer has {len(self.buffer)}/{self.seq_len} samples. Prediction not yet available.',
                'severity': 'INFO'
            }]

        return {
            'predictions': predictions,
            'max_predicted': np.nanmax(predictions) if not np.isnan(predictions).all() else None,
            'min_predicted': np.nanmin(predictions) if not np.isnan(predictions).all() else None,
            'alerts': alerts,
            'buffer_size': len(self.buffer)
        }

    def get_route_risk_assessment(self, route_points):
        """
        Assess risk for entire route based on predicted conditions.
        route_points: list of dicts each with 'sensor_data', 'remote_sensing_data', 'location', 'timestamp'
        Returns list of risk assessments per point.
        """
        # Reset buffer for new route
        self.buffer = []
        risk_scores = []

        for point in route_points:
            result = self.step(point['sensor_data'], point['remote_sensing_data'])

            # Calculate risk score (0-100) based on predictions
            if result['max_predicted'] is not None:
                temp_risk = max(0, (result['max_predicted'] - self.threshold)) * 20
                temp_risk = min(100, temp_risk)
                trend_risk = 20 if any(a['type'] == 'RAPID_TEMP_RISE' for a in result['alerts']) else 0
                risk_score = min(100, temp_risk + trend_risk)
            else:
                risk_score = 0

            risk_scores.append({
                'location': point['location'],
                'timestamp': point['timestamp'],
                'risk_score': risk_score,
                'max_temp': result['max_predicted'],
                'alerts': result['alerts']
            })

        return risk_scores


# Example usage
if __name__ == "__main__":
    # Configuration
    INPUT_DIM = 30  # Must match training: number of features after preprocessing
    SEQ_LEN = 12
    PRED_HORIZON = 6
    THRESHOLD = 6.0  # Celsius

    # Initialize monitor (choose appropriate model class)
    monitor = ColdChainMonitor(
        model_path='bilstm_attention_final.pth',  # your saved .pth file
        model_class=BiLSTMAttention,  # or KLSTM
        preprocessor_path='preprocessor.pkl',  # saved preprocessor
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        prediction_horizon=PRED_HORIZON,
        threshold_c=THRESHOLD
    )

    # Simulate a new sensor reading
    current_sensor = {
        'internal_temperature_c': 4.2,
        'ambient_temperature_c': 22.5,
        'humidity_percent': 68,
        'door_open': 0,
        'vibration_level': 0.3,
        'latitude': 40.7580,
        'longitude': -73.9855,
        'timestamp': datetime.now()
    }

    remote_sensing = {
        'satellite_ambient_temp': 23.1,
        'solar_radiation': 450,
        'land_surface_temp': 26.4,
        'elevation_m': 10,
        'ndvi': 0.15,
        'ndbi': 0.32
    }

    # Process the reading
    result = monitor.step(current_sensor, remote_sensing)

    print("=== Cold Chain Monitor Output ===")
    print(f"Buffer size: {result['buffer_size']}/{SEQ_LEN}")
    print(f"Predictions (next {PRED_HORIZON} timesteps): {result['predictions']}")
    print(f"Max predicted: {result['max_predicted']}")
    print(f"Alerts: {len(result['alerts'])}")
    for alert in result['alerts']:
        print(f"  - [{alert['severity']}] {alert['message']}")