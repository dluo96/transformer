import math
from typing import Any, Tuple

import torch
import torch.nn as nn

from learning.transformer.utils import clones


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: torch.float32 = None,
) -> Tuple[torch.Tensor, Any]:
    """Compute the 'scaled dot-product attention' for one attention head.

    Returns:
        - The updated embedding after applying the attention to the value.
            Has shape (num_batches, h, n, d_k).
        - The attention matrix. Shape (num_batches, h, n, n).
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """Compute the multihead attention. Due to the reduced embedding dimensionality
        of each head, the total computational cost is similar to that of single-head
        attention with full dimensionality. The time complexity of each attention head
        is O(n^2 * d_k), so the full layer has complexity O(n^2 * d) since d_k = d/h.

        Args:
            h: number of heads.
            d_model: embedding dimensionality.
            dropout: probability of an element being zeroed.
        """
        super(MultiHeadedAttention, self).__init__()

        # Reduce the dimensionality of each head
        assert d_model % h == 0
        self.d_k = d_model // h  # NB: we assume d_v always equals d_k
        self.h = h

        # Create 4 linear layers - 3 for the linear projections of Q, K, V
        # and 1 for the final linear layer in the multihead attention block
        self.linear_layers = clones(nn.Linear(d_model, d_model), 4)  # Not d_k
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """The multihead attention consists of `h` attention layers running
        in parallel."""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        num_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear layer.
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.h * self.d_k)
        del query, key, value
        final_linear_layer = self.linear_layers[-1]
        return final_linear_layer(x)


if __name__ == "__main__":
    n = 11
    d = 512
    h = 8
    d_k = 512 // h
    num_batches = 2
    Q = torch.rand(num_batches, n, 512)
    K = torch.rand(num_batches, n, 512)
    V = torch.rand(num_batches, n, 512)

    mha = MultiHeadedAttention(h, d)
    out = mha(Q, K, V)
