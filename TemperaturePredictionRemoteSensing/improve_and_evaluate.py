"""
improve_and_evaluate.py  –  Comprehensive Evaluation of Pre-Trained Cold Chain Models
===================================================================================
This script:
  1. Loads the synthetic dataset from 'cold_chain_dataset.csv'.
  2. Loads the saved preprocessor and sequences the data.
  3. Loads the pre-trained K-LSTM and BiLSTM-Attention models (no re-training!).
  4. Evaluates models using expanded metrics:
     - RMSE, MAE, MAPE, R² Score
     - SMAPE (Symmetric MAPE)
     - Explained Variance Score
     - Median Absolute Error
  5. Saves results to 'evaluation_results.csv' and generates plotting charts.
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, explained_variance_score,
    mean_absolute_percentage_error
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from K_LSTM_Model import KLSTM
from BiLSTM_Attention_Model import BiLSTMAttention

# ─────────────────────────────────────────────────────────────────────────────
# 1 · Configuration & Data Loading
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN       = 12
PRED_HORIZON  = 6
INPUT_DIM     = 14
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n{'='*60}")
print(f"  Model Performance Evaluation (Inference Only)")
print(f"  Device : {DEVICE.upper()}")
print(f"{'='*60}\n")

print("[1/4] Loading dataset and preprocessor ...")

csv_path = 'cold_chain_dataset.csv'
if not os.path.exists(csv_path):
    print(f"❌ ERROR: {csv_path} not found. Run generate_dataset.py.")
    sys.exit(1)

df = pd.read_csv(csv_path)

if not os.path.exists('preprocessor.pkl'):
    print("❌ ERROR: preprocessor.pkl not found! Run run_demo.py first.")
    sys.exit(1)

class SimplePreprocessor:
    def __init__(self, feature_scaler, target_scaler, feature_columns):
        self.feature_scaler  = feature_scaler
        self.target_scaler   = target_scaler
        self.feature_columns = feature_columns

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Extract features used during training
feature_names = [
    'internal_temperature_c', 'ambient_temperature_c', 'humidity_percent',
    'door_open', 'vibration_level', 'latitude', 'longitude',
    'satellite_ambient_temp', 'solar_radiation', 'land_surface_temp',
    'elevation_m', 'ndvi', 'ndbi', 'hour_sin'
]

raw_data = df[feature_names].values
internal_temp = df['internal_temperature_c'].values.reshape(-1, 1)

# Ensure preprocessor scales raw_data correctly
X_scaled = preprocessor.feature_scaler.transform(raw_data)
y_scaled = preprocessor.target_scaler.transform(internal_temp)

def make_sequences(X, y, seq_len, pred_horizon):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_horizon + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + pred_horizon, 0])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_HORIZON)

# Same split as training (70/15/15) to get the held-out test set
n = len(X_seq)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
X_test, y_test_scaled_seq = X_seq[n_train+n_val:], y_seq[n_train+n_val:]

print(f"   Test Set Shape: X={X_test.shape}, y={y_test_scaled_seq.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Load Pre-Trained Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Loading pre-trained PyTorch models ...")
models = {}

# BiLSTM Attention
if os.path.exists('best_bilstm_attention.pth'):
    m = BiLSTMAttention(input_dim=INPUT_DIM, prediction_horizon=PRED_HORIZON).to(DEVICE)
    m.load_state_dict(torch.load('best_bilstm_attention.pth', map_location=DEVICE, weights_only=True))
    m.eval()
    models['BiLSTM-Attention'] = m
else:
    print("⚠️ best_bilstm_attention.pth not found.")

# K-LSTM Finetuned
if os.path.exists('best_klstm_finetuned.pth'):
    m = KLSTM(input_dim=INPUT_DIM, seq_len=SEQ_LEN, prediction_horizon=PRED_HORIZON).to(DEVICE)
    m.load_state_dict(torch.load('best_klstm_finetuned.pth', map_location=DEVICE, weights_only=True))
    m.eval()
    models['K-LSTM (Finetuned)'] = m
elif os.path.exists('best_klstm.pth'):
    m = KLSTM(input_dim=INPUT_DIM, seq_len=SEQ_LEN, prediction_horizon=PRED_HORIZON).to(DEVICE)
    m.load_state_dict(torch.load('best_klstm.pth', map_location=DEVICE, weights_only=True))
    m.eval()
    models['K-LSTM (Frozen)'] = m
else:
    print("⚠️ best_klstm.pth not found.")

# Backup: _final.pth versions if best are missing
if not models:
    for name, class_ref in [('bilstm_attention_final.pth', BiLSTMAttention), ('klstm_finetuned_final.pth', KLSTM), ('klstm_final.pth', KLSTM)]:
        if os.path.exists(name):
            if name.startswith('bilstm'):
                m = class_ref(input_dim=INPUT_DIM, prediction_horizon=PRED_HORIZON).to(DEVICE)
            else:
                m = class_ref(input_dim=INPUT_DIM, seq_len=SEQ_LEN, prediction_horizon=PRED_HORIZON).to(DEVICE)
            m.load_state_dict(torch.load(name, map_location=DEVICE, weights_only=True))
            m.eval()
            models[name.replace('_final.pth', '')] = m

if not models:
    print("❌ No models found to evaluate! Please run run_demo.py first.")
    sys.exit(1)

for name in models:
    print(f"   Loaded: {name}")

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Computing Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Running inference and computing metrics ...")

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

X_test_t = torch.FloatTensor(X_test).to(DEVICE)
y_test_orig = preprocessor.target_scaler.inverse_transform(y_test_scaled_seq)
y_test_flat = y_test_orig.flatten()

metrics_list = []
predictions_orig = {}

for name, model in models.items():
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()
        
    y_pred_orig = preprocessor.target_scaler.inverse_transform(y_pred_scaled)
    predictions_orig[name] = y_pred_orig
    y_pred_flat = y_pred_orig.flatten()
    
    rmse  = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
    mae   = mean_absolute_error(y_test_flat, y_pred_flat)
    r2    = r2_score(y_test_flat, y_pred_flat)
    mape  = mean_absolute_percentage_error(y_test_flat, y_pred_flat) * 100
    smape = symmetric_mean_absolute_percentage_error(y_test_flat, y_pred_flat)
    med_ae= median_absolute_error(y_test_flat, y_pred_flat)
    ev    = explained_variance_score(y_test_flat, y_pred_flat)
    
    metrics_list.append({
        'Model': name,
        'RMSE (°C)': rmse,
        'MAE (°C)': mae,
        'Median AE': med_ae,
        'MAPE (%)': mape,
        'SMAPE (%)': smape,
        'R² Score': r2,
        'Explained Var': ev
    })

# Ensemble Evaluation
if len(models) > 1:
    ens_pred_orig = np.mean(list(predictions_orig.values()), axis=0)
    predictions_orig['Ensemble'] = ens_pred_orig
    ens_pred_flat = ens_pred_orig.flatten()
    
    metrics_list.append({
        'Model': 'Ensemble',
        'RMSE (°C)': np.sqrt(mean_squared_error(y_test_flat, ens_pred_flat)),
        'MAE (°C)': mean_absolute_error(y_test_flat, ens_pred_flat),
        'Median AE': median_absolute_error(y_test_flat, ens_pred_flat),
        'MAPE (%)': mean_absolute_percentage_error(y_test_flat, ens_pred_flat) * 100,
        'SMAPE (%)': symmetric_mean_absolute_percentage_error(y_test_flat, ens_pred_flat),
        'R² Score': r2_score(y_test_flat, ens_pred_flat),
        'Explained Var': explained_variance_score(y_test_flat, ens_pred_flat)
    })

metrics_df = pd.DataFrame(metrics_list)

csv_path = 'evaluation_results.csv'
metrics_df.to_csv(csv_path, index=False)
print("\n" + metrics_df.to_string(index=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Generate Visualisations
# ─────────────────────────────────────────────────────────────────────────────
print("[4/4] Generating visualisations ...")
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Expanded Model Performance Metrics', fontsize=16)

plots_config = [
    ('RMSE (°C)', 0, 0, 'Lower is better'),
    ('MAE (°C)', 0, 1, 'Lower is better'),
    ('Median AE', 0, 2, 'Lower is better'),
    ('SMAPE (%)', 1, 0, 'Lower is better'),
    ('R² Score', 1, 1, 'Higher is better'),
    ('Explained Var', 1, 2, 'Higher is better (Max 1.0)'),
]

for col, row, col_idx, subtitle in plots_config:
    sns.barplot(data=metrics_df, x='Model', y=col, ax=axes[row, col_idx], palette='magma')
    axes[row, col_idx].set_title(f'{col}\n({subtitle})')
    axes[row, col_idx].tick_params(axis='x', rotation=15)
    
axes[1, 1].set_ylim(bottom=max(-1, metrics_df['R² Score'].min() - 0.2), top=1.05)
axes[1, 2].set_ylim(bottom=max(-1, metrics_df['Explained Var'].min() - 0.2), top=1.05)

plt.tight_layout()
metrics_plot_path = 'improved_performance_metrics.png'
plt.savefig(metrics_plot_path, dpi=300)

# True vs Predicted (First step in horizon t+1)
plt.figure(figsize=(14, 6))
time_axis = np.arange(min(300, len(y_test_orig))) # Zoom in on 300 samples for clarity

plt.plot(time_axis, y_test_orig[:300, 0], label='True Temp (t+1)', color='black', linewidth=2, linestyle='--')
colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800']
for (name, preds), color in zip(predictions_orig.items(), colors):
    plt.plot(time_axis, preds[:300, 0], label=f'{name} Pred', color=color, alpha=0.8, linewidth=1.5)

plt.title('True vs Predicted Output Temperature (Held-out Test Set zoomed, t+1 step)')
plt.xlabel('Test Set Sample Index')
plt.ylabel('Internal Temperature (°C)')
plt.legend()
plt.tight_layout()
tvp_plot_path = 'improved_true_vs_predicted.png'
plt.savefig(tvp_plot_path, dpi=300)

print(f"\n✅ All tasks complete. Results saved in {csv_path}")
print(f"{'='*60}\n")
