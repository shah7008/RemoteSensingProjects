"""
run_demo.py  –  End-to-end demo for the Cold Chain Temperature Prediction project
===================================================================================
This script:
  1. Generates synthetic cold-chain sensor + remote-sensing data
  2. Builds a lightweight preprocessor (scaler + feature list)
  3. Pre-trains the Convolutional Autoencoder
  4. Trains the K-LSTM and BiLSTM-Attention models
  5. Evaluates on a held-out test set
  6. Saves all model files + preprocessor
  7. Runs a real-time monitoring simulation with a rolling buffer
"""

import os, sys, pickle
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ── make sure sibling modules are importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from K_LSTM_Model import KLSTM
from BiLSTM_Attention_Model import BiLSTMAttention
from Convolutional_Autoencoder import ConvAutoencoder
from Training_Pipeline import ColdChainPyTorchPipeline, EarlyStopping

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN       = 12          # input sequence length (time-steps)
PRED_HORIZON  = 6           # how many future steps to predict
INPUT_DIM     = 14          # number of input features
N_SAMPLES     = 2000        # synthetic data points
BATCH_SIZE    = 32
AE_EPOCHS     = 20          # autoencoder pre-training epochs
TRAIN_EPOCHS  = 40          # main model training epochs
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n{'='*60}")
print(f"  Cold Chain Temperature Prediction – Demo Run")
print(f"  Device : {DEVICE.upper()}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1 · Generate synthetic data
# ─────────────────────────────────────────────────────────────────────────────
print("[1/6] Loading cold-chain data from CSV …")
import pandas as pd

csv_path = 'cold_chain_dataset.csv'
if not os.path.exists(csv_path):
    print(f"❌ ERROR: {csv_path} not found. Please run generate_dataset.py first.")
    sys.exit(1)

df = pd.read_csv(csv_path)
feature_names = [
    'internal_temperature_c', 'ambient_temperature_c', 'humidity_percent',
    'door_open', 'vibration_level', 'latitude', 'longitude',
    'satellite_ambient_temp', 'solar_radiation', 'land_surface_temp',
    'elevation_m', 'ndvi', 'ndbi', 'hour_sin'
]

raw_data = df[feature_names].values
internal_temp = df['internal_temperature_c'].values
N_SAMPLES = len(df)

print(f"   Raw data shape : {raw_data.shape} (Loaded from CSV)")

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Build preprocessor and create sequences
# ─────────────────────────────────────────────────────────────────────────────
print("[2/6] Preprocessing data …")

feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(raw_data)
y_raw    = internal_temp.reshape(-1, 1)
y_scaled = target_scaler.fit_transform(y_raw)

def make_sequences(X, y, seq_len, pred_horizon):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_horizon + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + pred_horizon, 0])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_HORIZON)
print(f"   Sequence shapes  X={X_seq.shape}  y={y_seq.shape}")

# Train / val / test split  (70 / 15 / 15)
n = len(X_seq)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train, y_train = X_seq[:n_train],        y_seq[:n_train]
X_val,   y_val   = X_seq[n_train:n_train+n_val], y_seq[n_train:n_train+n_val]
X_test,  y_test  = X_seq[n_train+n_val:],  y_seq[n_train+n_val:]

print(f"   Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

# Persist a simple preprocessor object so the monitor can use it
class SimplePreprocessor:
    def __init__(self, feature_scaler, target_scaler, feature_columns):
        self.feature_scaler  = feature_scaler
        self.target_scaler   = target_scaler
        self.feature_columns = feature_columns

preprocessor = SimplePreprocessor(feature_scaler, target_scaler, feature_names)
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("   Saved preprocessor.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Pre-train the Convolutional Autoencoder
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Pre-training Convolutional Autoencoder …")
pipeline = ColdChainPyTorchPipeline(INPUT_DIM, SEQ_LEN, PRED_HORIZON, device=DEVICE)
autoencoder = pipeline.pretrain_autoencoder(X_train, epochs=AE_EPOCHS, batch_size=BATCH_SIZE)
print("   Autoencoder pre-training done.")

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Train K-LSTM and BiLSTM-Attention
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Training models ...")
train_loader, val_loader = pipeline.create_dataloaders(
    X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE
)

print("  >> K-LSTM (frozen encoder) ...")
history_klstm, history_klstm_ft = pipeline.train_klstm(
    train_loader, val_loader,
    autoencoder_weights_path='autoencoder_pretrained.pth'
)

print("  >> BiLSTM-Attention ...")
history_bilstm = pipeline.train_bilstm_attention(train_loader, val_loader)

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Evaluating on test set ...")
X_test_t = torch.FloatTensor(X_test).to(DEVICE)
y_test_np = y_test                        # numpy (n_test, pred_horizon)

results_table = []
for name, model in pipeline.models.items():
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()
    rmse = np.sqrt(np.mean((y_pred - y_test_np) ** 2))
    results_table.append((name, rmse))
    print(f"   {name:<30} Test RMSE (scaled): {rmse:.6f}")

ensemble_pred = pipeline.ensemble_predict(X_test)
ens_rmse = np.sqrt(np.mean((ensemble_pred - y_test_np) ** 2))
results_table.append(('ensemble', ens_rmse))
print(f"   {'ensemble':<30} Test RMSE (scaled): {ens_rmse:.6f}")

best_model_name = min(results_table, key=lambda x: x[1])[0]
print(f"\n   Best model: {best_model_name}")

# ─────────────────────────────────────────────────────────────────────────────
# 6 · Run real-time monitoring simulation (no saved .pth needed here)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Real-time monitoring simulation …")

# Use the best model already in memory
monitor_model = pipeline.models.get(best_model_name)
if monitor_model is None:
    monitor_model = list(pipeline.models.values())[0]
monitor_model.eval()

# Rolling buffer simulation
buffer = []

def preprocess_reading(sensor, remote_sensing):
    """Combine sensor + remote-sensing into a scaled feature vector."""
    combined = {**sensor, **remote_sensing}
    vec = np.array([
        combined.get('internal_temperature_c', 4.0),
        combined.get('ambient_temperature_c', 22.0),
        combined.get('humidity_percent', 65.0),
        combined.get('door_open', 0.0),
        combined.get('vibration_level', 0.3),
        combined.get('latitude', 40.75),
        combined.get('longitude', -73.98),
        combined.get('satellite_ambient_temp', 22.5),
        combined.get('solar_radiation', 400.0),
        combined.get('land_surface_temp', 26.0),
        combined.get('elevation_m', 10.0),
        combined.get('ndvi', 0.15),
        combined.get('ndbi', 0.32),
        np.sin(2 * np.pi * datetime.now().hour / 24),
    ], dtype=np.float32)
    return preprocessor.feature_scaler.transform(vec.reshape(1, -1))[0]

def predict_from_buffer(buf):
    """Build sequence from buffer, run model, inverse-transform predictions."""
    if len(buf) >= SEQ_LEN:
        seq = np.array([b for b in buf[-SEQ_LEN:]], dtype=np.float32)
    else:
        pad = [buf[0]] * (SEQ_LEN - len(buf))
        seq = np.array(pad + buf, dtype=np.float32)
    X_t = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y_pred_scaled = monitor_model(X_t).cpu().numpy().flatten()
    y_pred_orig = preprocessor.target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()
    return y_pred_orig

THRESHOLD = 6.0  # °C

print(f"\n  {'Step':<6} {'Buffer':>8} {'MaxPred':>10} {'Alert'}")
print(f"  {'-'*50}")

base_time = datetime.now()
for step in range(20):
    ts = base_time + timedelta(minutes=step * 30)

    # Simulate slightly rising temperature after step 14
    temp = 4.0 + 0.1 * step + 0.3 * np.sin(step) + 0.2 * np.random.randn()

    sensor_data = {
        'internal_temperature_c': temp,
        'ambient_temperature_c' : 22.0 + np.random.randn(),
        'humidity_percent'       : 65.0 + 5 * np.random.randn(),
        'door_open'              : 1.0 if step == 10 else 0.0,
        'vibration_level'        : 0.3 + 0.05 * np.random.rand(),
        'latitude'               : 40.758,
        'longitude'              : -73.985,
        'timestamp'              : ts,
    }
    remote_data = {
        'satellite_ambient_temp': 23.0 + np.random.randn(),
        'solar_radiation'        : 450 + 30 * np.random.randn(),
        'land_surface_temp'      : 26.0 + np.random.randn(),
        'elevation_m'            : 10.0,
        'ndvi'                   : 0.15,
        'ndbi'                   : 0.32,
    }

    feat = preprocess_reading(sensor_data, remote_data)
    buffer.append(feat)

    predictions = predict_from_buffer(buffer)
    max_pred    = np.max(predictions)

    alert_msg = ''
    if max_pred > THRESHOLD:
        sev = 'CRITICAL' if max_pred > THRESHOLD + 2 else 'WARNING'
        alert_msg = f'[{sev}] HIGH_TEMP_RISK {max_pred:.1f}°C > {THRESHOLD}°C'
    diffs = np.diff(predictions)
    if np.any(diffs > 1.0):
        alert_msg += (' | ' if alert_msg else '') + f'[WARNING] RAPID_RISE +{np.max(diffs):.2f}°C'

    print(f"  {step+1:<6} {len(buffer):>8} {max_pred:>10.2f}  {alert_msg or 'OK'}")

print(f"\n{'='*60}")
print("  Demo complete! All model files saved in:")
print(f"  {os.path.abspath('.')}")
print(f"{'='*60}\n")
