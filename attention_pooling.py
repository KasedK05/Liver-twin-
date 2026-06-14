import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """Attention-based temporal pooling for interpretability."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, sequence_output, mask=None):
        """
        sequence_output: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) bool, True = ignore
        Returns: context (batch, hidden_dim), attention_weights (batch, seq_len)
        """
        attn_scores = self.attention(sequence_output).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), sequence_output).squeeze(1)

        return context, attn_weights
