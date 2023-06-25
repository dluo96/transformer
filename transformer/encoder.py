import torch
from torch import nn

from transformer import (
    LayerNorm,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    SublayerConnection,
    clones,
)


class EncoderLayer(nn.Module):
    """An encoder layer consists of two sublayers:
        - Self-attention + residual connection followed by layer norm.
        - Feedforward NN + residual connection followed by layer norm.

    The `size` is typically the embedding dimensionality.
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pass the input through self-attention + residual connection, then
        through feedforward neural network + residual connection.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """The encoder is a stack of N encoder layers, followed by a layer norm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pass the input (and mask) through each encoder layer in turn, then
        pass the result of this to a layer norm. The result is the final output
        of the encoder in the transformer architecture."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
