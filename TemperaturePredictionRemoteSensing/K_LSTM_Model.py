import torch
import torch.nn as nn
from Convolutional_Autoencoder import ConvAutoencoder

class KLSTM(nn.Module):
    """
    K-LSTM: Pre-trained convolutional autoencoder + LSTM with attention
    """

    def __init__(self, input_dim, seq_len, hidden_dim=64, num_layers=2,
                 encoding_dim=32, prediction_horizon=6):
        super(KLSTM, self).__init__()

        # Autoencoder (pre-trained separately)
        self.autoencoder = ConvAutoencoder(input_dim, seq_len, encoding_dim)
        self.encoder = self.autoencoder.encoder  # reuse encoder part
        self.encoder_fc = self.autoencoder.encoder_fc

        # Freeze encoder initially
        self.freeze_encoder()

        # LSTM with attention
        self.lstm = nn.LSTM(
            input_size=encoding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        print("KLSTM called InIt...")

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, prediction_horizon)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder_fc.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.encoder_fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # Get encoded features
        x_enc = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        encoded = self.encoder(x_enc)  # (batch, 16, 1)
        encoded = encoded.squeeze(-1)  # (batch, 16)
        encoded = self.encoder_fc(encoded)  # (batch, encoding_dim)

        # Repeat encoded features across time steps for LSTM input
        # We need a sequence for LSTM; one approach: repeat the same vector
        repeated = encoded.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, encoding_dim)

        # LSTM
        lstm_out, _ = self.lstm(repeated)  # (batch, seq_len, hidden_dim)

        # Attention: compute weights for each time step
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)

        # Apply attention
        attended = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)

        # Fully connected layers
        x = self.relu(self.fc1(attended))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)  # (batch, prediction_horizon)

        return out