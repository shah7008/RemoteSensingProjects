# pytorch_models.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class ConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for feature extraction
    Used in K-LSTM architecture
    """

    def __init__(self, input_dim, seq_len, encoding_dim=32):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # Global max pooling
        )

        # The output of encoder will be (batch, 16, 1) -> flatten to (batch, 16)
        # Then project to encoding_dim
        self.encoder_fc = nn.Linear(16, encoding_dim)

        # Decoder
        self.decoder_fc = nn.Linear(encoding_dim, 16 * (seq_len // 4))  # Adjust based on pooling
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(64, input_dim, kernel_size=3, padding=1)
        )

        self.seq_len = seq_len
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> Conv1d expects (batch, input_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        # Encode
        encoded = self.encoder(x)  # (batch, 16, 1)
        encoded = encoded.squeeze(-1)  # (batch, 16)
        encoded = self.encoder_fc(encoded)  # (batch, encoding_dim)

        # Decode
        decoded = self.decoder_fc(encoded)  # (batch, 16 * (seq_len//4))
        decoded = decoded.view(-1, 16, self.seq_len // 4)  # (batch, 16, seq_len//4)
        decoded = self.decoder(decoded)  # (batch, input_dim, seq_len)

        # Return to original shape (batch, seq_len, input_dim)
        decoded = decoded.transpose(1, 2)
        return decoded, encoded