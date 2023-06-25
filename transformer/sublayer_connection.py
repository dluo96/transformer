from typing import Union

import torch
from torch import nn

from transformer import LayerNorm, MultiHeadedAttention, PositionwiseFeedForward


class SublayerConnection(nn.Module):
    """A residual connection followed by a `LayerNorm`.
    NOTE: for code simplicity the layer norm is first rather than last.
    """

    def __init__(self, size: int, dropout: float) -> None:
        """The `size` is typically the dimensionality of the embedding space. The
        `dropout` is the probability of an element being zeroed.
        """
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Union[MultiHeadedAttention, PositionwiseFeedForward],
    ) -> torch.Tensor:
        """Apply residual connection to a sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
