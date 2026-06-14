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
        position = t.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=t.device) * 
                            (-math.log(10000.0) / self.d_model))

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
