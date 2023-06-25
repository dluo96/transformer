import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed forward neural network consisting of two linear transformations,
    with dropout regularization applied. WHAT DOES POSITION WISE MEAN?
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform two linear transformations with a ReLU activation function
        in between as well as dropout."""
        return self.w_2(self.dropout(self.w_1(x).relu()))
