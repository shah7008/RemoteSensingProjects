# pytorch_training_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pytorch_models import KLSTM, BiLSTMAttention, ConvAutoencoder


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ColdChainPyTorchPipeline:
    def __init__(self, input_dim, seq_len, prediction_horizon, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.device = device
        self.models = {}

    def pretrain_autoencoder(self, X_train, epochs=50, batch_size=32, lr=1e-3):
        """Pre-train convolutional autoencoder for feature extraction"""
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        autoencoder = ConvAutoencoder(self.input_dim, self.seq_len).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

        autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                reconstructed, _ = autoencoder(x)
                loss = criterion(reconstructed, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.6f}")

        # Save encoder weights for later use
        torch.save(autoencoder.state_dict(), 'autoencoder_pretrained.pth')
        return autoencoder

    def create_dataloaders(self, X_train, y_train, X_val, y_val, batch_size=32):
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(self, model, train_loader, val_loader, epochs=100, lr=1e-3, model_name='model'):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=15)

        model.to(self.device)

        history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

                    all_preds.append(y_pred.cpu().numpy())
                    all_targets.append(y_batch.cpu().numpy())

            val_loss /= len(val_loader.dataset)

            # Calculate RMSE
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_{model_name}.pth')
                print(f"  Best model saved (val_loss: {val_loss:.4f})")
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_rmse'].append(rmse)

            scheduler.step(val_loss)
            early_stopping(val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_rmse: {rmse:.4f}")

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.models[model_name] = model
        torch.save(model.state_dict(), f'{model_name}_final.pth')

        return history

    def train_klstm(self, train_loader, val_loader, autoencoder_weights_path='autoencoder_pretrained.pth'):
        """Train K-LSTM with pre-trained autoencoder"""
        # Build model
        model = KLSTM(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            prediction_horizon=self.prediction_horizon
        )

        # Load pre-trained autoencoder weights
        autoencoder_state = torch.load(autoencoder_weights_path)
        model.autoencoder.load_state_dict(autoencoder_state)
        model.freeze_encoder()  # Freeze encoder layers

        print("Training K-LSTM (encoder frozen)...")
        history = self.train_model(model, train_loader, val_loader,
                                   epochs=100, lr=1e-3, model_name='klstm')

        # Optional: fine-tune entire model
        print("Fine-tuning K-LSTM (unfreeze encoder)...")
        model.unfreeze_encoder()
        history_ft = self.train_model(model, train_loader, val_loader,
                                      epochs=50, lr=1e-4, model_name='klstm_finetuned')

        return history, history_ft

    def train_bilstm_attention(self, train_loader, val_loader):
        model = BiLSTMAttention(
            input_dim=self.input_dim,
            prediction_horizon=self.prediction_horizon
        )
        history = self.train_model(model, train_loader, val_loader,
                                   epochs=100, lr=1e-3, model_name='bilstm_attention')
        return history

    def ensemble_predict(self, X, model_names=None):
        """Make predictions using ensemble of trained models"""
        if model_names is None:
            model_names = list(self.models.keys())

        if not model_names:
            raise ValueError("No models available")

        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []

        for name in model_names:
            model = self.models[name]
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
            predictions.append(pred)

        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred