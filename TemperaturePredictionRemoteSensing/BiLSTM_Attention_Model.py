import torch
import torch.nn as nn

class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, prediction_horizon=6, dropout=0.4):
        super(BiLSTMAttention, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        print("BiLSTMAttention called InIt...")

        # Attention mechanism (on concatenated forward+backward hidden states)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, prediction_horizon)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)

        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        attended = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)

        # Fully connected
        x = self.relu(self.bn1(self.fc1(attended)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        out = self.fc3(x)

        return out