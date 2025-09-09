import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import BASE

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as in "Attention Is All You Need".
    Adds positional information to token embeddings.
    """

    def __init__(self, d_model = BASE['d_model'], dropout=BASE['dropout'], max_len=5000):
        """
        Args:
            d_model: Embedding dimension (from configs.py)
            dropout: Dropout rate (from configs.py)
            max_len: Maximum sequence length to precompute
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        # apply sin to even indices
        pe[:, 0::2] = torch.sin(position*div_term)
        
        # apply cos to odd indices
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        """
            Args:
                x: Input embeddings of shape (batch, seq_len, d_model)
            Returns:
                Embeddings + positional encodings with droupout
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.droupout(x)

