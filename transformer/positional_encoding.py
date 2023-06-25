import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the embedding for each word
        in the input sequence.

        Args:
            x: input sequence embeddings of shape (B, N, D) where
                B is the batch size, N is the length of the input
                sequence, and D is the embedding dimensionality
                i.e. d_model.

        Returns:
            Tensor of shape (B, N, D).
        """
        # Compute length of input sequence
        n = x.size(1)

        # Only need positional encodings for the first n positions
        x = x + self.pe[:, :n].requires_grad_(False)
        return self.dropout(x)


if __name__ == "__main__":
    pos_enc = PositionalEncoding(512, 0.5)
    y = torch.rand(1, 11, 512)
    z = pos_enc(y)
