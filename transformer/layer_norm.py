import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, num_features: int, epsilon: torch.float32 = 1e-6) -> None:
        """The `num_features` is typically the dimensionality of the embedding space.
        Importantly, the condition x.shape[-1] == num_features must be satisfied where
        x is the input to `forward` below.
        """
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """For each element (e.g. word) of the input sequence, compute its mean and
        standard deviation across all of its values in the embedding space. Use these
        to scale its embedding to have zero mean and unit variance. Finally, the
        learnable parameters a_2 and b_2 allow adjustment of distributions. Crucially,
        a_2*(x-mean) is an element-wise multiplication and thus can be used to change
        the effective standard deviation, and b_2 is the effective mean. By default,
        a_2 and b_2 are initialised to correspond with Z-score normalization (aka
        standardization).
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.epsilon) + self.b_2
