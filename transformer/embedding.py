import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class Embeddings(nn.Module):
    """In our model, we share the same weight matrix between the two embedding layers."""

    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """Does not work when x is `torch.Tensor`"""
        return self.lookup_table(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    """Defines a standard linear + softmax generation step. The linear
    layer projects its input back to a space where we can extract the
    exact word (de-embedding if you will). This extraction is done via
    a log softmax (used instead of a softmax for numerical instability,
    optimisation, and heavy penalty for highly incorrect predictions).
    """

    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output probabilities for each word in vocabulary."""
        return log_softmax(self.proj(x), dim=-1)
