"""Module for defining the decoder part of the transformer architecture."""
import torch
from torch import nn

from transformer.attention import MultiHeadedAttention
from transformer.feedforward_net import PositionwiseFeedForward
from transformer.layer_norm import LayerNorm
from transformer.sublayer_connection import SublayerConnection
from transformer.utils import clones


class DecoderLayer(nn.Module):
    """Each decoder layer consists of three sublayers:
    - Masked multihead attention + residual connection followed by layer norm.
        Denoted `self_attn` below.
    - Multihead attention + residual connection followed by layer norm. The
        input is the keys and values output by the encoder, and the query from
        the output of the masked multihead attention. Denoted `src_attn` below.
    - Feedforward neural network + residual connection followed by a layer norm.
        Denoted `feed_forward` below.
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        # Recall there are 3 sublayers
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """The decoder is a stack of N decoder layers, and it can contain masking."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pass the input (and mask) through each decoder layer in turn, then
        pass the result of this to a layer norm. The result is the final output
        of the decoder.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
