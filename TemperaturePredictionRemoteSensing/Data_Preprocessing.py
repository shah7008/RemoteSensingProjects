# main_pytorch.py
import pandas as pd
import numpy as np
# from Data_Preprocessing import ColdChainDataPreprocessor
from Training_Pipeline import ColdChainPyTorchPipeline
import torch
# Load and preprocess data (same as before)
preprocessor = ColdChainDataPreprocessor(sequence_length=12, prediction_horizon=6)
df = pd.read_csv('enriched_cold_chain_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
data_dict = preprocessor.prepare_data(df, target_col='internal_temperature_c')

# ---------- SAVE PREPROCESSOR ----------
import pickle
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
# ----------------------------------------
print("DataPreprocessor called...")
X_train, y_train = data_dict['X_train'], data_dict['y_train']
X_val, y_val = data_dict['X_val'], data_dict['y_val']
X_test, y_test = data_dict['X_test'], data_dict['y_test']

# Initialize PyTorch pipeline
input_dim = X_train.shape[2]
seq_len = X_train.shape[1]
pred_horizon = y_train.shape[1]

pipeline = ColdChainPyTorchPipeline(input_dim, seq_len, pred_horizon)

# Pre-train autoencoder
autoencoder = pipeline.pretrain_autoencoder(X_train, epochs=50)

# Create DataLoaders
train_loader, val_loader = pipeline.create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

# Train K-LSTM
history_klstm, history_klstm_ft = pipeline.train_klstm(train_loader, val_loader)

# Train BiLSTM-Attention
history_bilstm = pipeline.train_bilstm_attention(train_loader, val_loader)

# Evaluate on test set
X_test_tensor = torch.FloatTensor(X_test).to(pipeline.device)
y_test_tensor = torch.FloatTensor(y_test).to(pipeline.device)

for name, model in pipeline.models.items():
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    y_test_np = y_test
    rmse = np.sqrt(np.mean((y_pred - y_test_np)**2))
    print(f"{name} test RMSE: {rmse:.4f}")

# Ensemble prediction
ensemble_pred = pipeline.ensemble_predict(X_test)
ensemble_rmse = np.sqrt(np.mean((ensemble_pred - y_test)**2))
print(f"Ensemble test RMSE: {ensemble_rmse:.4f}")