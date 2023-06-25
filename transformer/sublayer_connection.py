"""Module for defining the sublayer connection."""
from __future__ import annotations

import torch
from torch import nn

from transformer.attention import MultiHeadedAttention
from transformer.feedforward_net import PositionwiseFeedForward
from transformer.layer_norm import LayerNorm


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
        sublayer: MultiHeadedAttention | PositionwiseFeedForward,
    ) -> torch.Tensor:
        """Apply residual connection to a sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
