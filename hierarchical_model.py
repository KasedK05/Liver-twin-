
"""
Hierarchical Cell Fate Prediction Model
Supports: Small LSTM, Big LSTM (BiLSTM), Transformer Encoder
Multi-task: Overall class (4-way) + Mechanism class (17-way)
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Continuous-time positional encoding for biological time series."""
    def __init__(self, d_model, max_time=24.0):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time

    def forward(self, t):
        """
        t: (batch, seq_len) actual time values in hours
        Returns: (batch, seq_len, d_model)
        """
        pe = torch.zeros(t.shape[0], t.shape[1], self.d_model, device=t.device)
        position = t.unsqueeze(-1)  # (batch, seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=t.device) * 
                            (-math.log(10000.0) / self.d_model))

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class HierarchicalCellModel(nn.Module):
    """
    Multi-task model for cell fate prediction.

    Architecture options:
      - 'lstm_small': 2-layer LSTM, 128 hidden
      - 'lstm_large': 4-layer BiLSTM, 256 hidden  
      - 'transformer': 4-layer Transformer encoder, 128 d_model
    """

    def __init__(self, 
                 input_dim=32,
                 architecture='lstm_small',
                 num_overall_classes=4,
                 num_mechanism_classes=17,
                 dropout=0.2):
        super().__init__()

        self.architecture = architecture
        self.input_dim = input_dim

        # --- ENCODER ---
        if architecture == 'lstm_small':
            self.hidden_dim = 128
            self.num_layers = 2
            self.encoder = nn.LSTM(
                input_dim, self.hidden_dim, self.num_layers,
                batch_first=True, dropout=dropout, bidirectional=False
            )
            self.encoder_output_dim = self.hidden_dim

        elif architecture == 'lstm_large':
            self.hidden_dim = 256
            self.num_layers = 4
            self.encoder = nn.LSTM(
                input_dim, self.hidden_dim, self.num_layers,
                batch_first=True, dropout=dropout, bidirectional=True
            )
            self.encoder_output_dim = self.hidden_dim * 2  # bidirectional

        elif architecture == 'transformer':
            self.d_model = 128
            self.nhead = 4
            self.num_layers = 4
            self.dim_feedforward = 512

            self.input_proj = nn.Linear(input_dim, self.d_model)
            self.pos_encoder = SinusoidalPositionalEncoding(self.d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.encoder_output_dim = self.d_model

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # --- POOLING ---
        # Use attention-based pooling for better interpretability
        self.attention_pool = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

        # --- MULTI-TASK HEADS ---
        # Head 1: Overall cell fate (4 classes)
        self.overall_head = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_overall_classes)
        )

        # Head 2: Mechanism subclass (17 classes)
        self.mechanism_head = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_mechanism_classes)
        )

        # Head 3: Subsystem health trajectory prediction (regression)
        # Predicts final health score for each of 7 subsystems
        self.subsystem_head = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7)  # 7 subsystem health scores
        )

    def forward(self, x, times, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim) trajectory
            times: (batch, seq_len) actual time values
            mask: (batch, seq_len) bool mask for padding

        Returns:
            overall_logits: (batch, num_overall_classes)
            mechanism_logits: (batch, num_mechanism_classes)
            subsystem_preds: (batch, 7)
            attention_weights: (batch, seq_len) for interpretability
        """
        batch_size, seq_len, _ = x.shape

        # --- ENCODE ---
        if self.architecture in ['lstm_small', 'lstm_large']:
            # LSTM encoding
            lstm_out, (h_n, c_n) = self.encoder(x)  # lstm_out: (batch, seq_len, hidden)

            # Attention pooling over time steps
            attn_scores = self.attention_pool(lstm_out).squeeze(-1)  # (batch, seq_len)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)

            # Weighted sum
            context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)

        elif self.architecture == 'transformer':
            # Project input
            x_proj = self.input_proj(x)  # (batch, seq_len, d_model)

            # Add positional encoding
            pos_enc = self.pos_encoder(times)
            x_proj = x_proj + pos_enc

            # Transformer encoding
            if mask is not None:
                # Create key padding mask for transformer (True = ignore)
                key_mask = mask  # (batch, seq_len)
                trans_out = self.encoder(x_proj, src_key_padding_mask=key_mask)
            else:
                trans_out = self.encoder(x_proj)

            # Attention pooling
            attn_scores = self.attention_pool(trans_out).squeeze(-1)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=1)

            context = torch.bmm(attn_weights.unsqueeze(1), trans_out).squeeze(1)

        # --- PREDICT ---
        overall_logits = self.overall_head(context)
        mechanism_logits = self.mechanism_head(context)
        subsystem_preds = self.subsystem_head(context)

        return overall_logits, mechanism_logits, subsystem_preds, attn_weights
