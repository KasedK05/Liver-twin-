import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StandardLSTM(nn.Module):
    """2-layer LSTM, 128 hidden units"""
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, 
                 num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len=49, input_dim=32)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        return self.fc(last_hidden)


class LargeLSTM(nn.Module):
    """4-layer bidirectional LSTM, 256 hidden units"""
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=4,
                 num_classes=6, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # Concatenate final forward and backward hidden states
        # h_n: (num_layers*2, batch, hidden_dim)
        forward = h_n[-2]   # (batch, hidden_dim)
        backward = h_n[-1]  # (batch, hidden_dim)
        combined = torch.cat([forward, backward], dim=1)
        return self.fc(combined)


class TransformerClassifier(nn.Module):
    """Transformer encoder with multi-head attention"""
    def __init__(self, input_dim=32, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len=49, input_dim=32)
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)          # (batch, seq_len, d_model)
        x = x.mean(dim=1)                # Global average pooling
        return self.fc(x)


MODEL_REGISTRY = {
    "lstm": StandardLSTM,
    "large_lstm": LargeLSTM,
    "transformer": TransformerClassifier,
}