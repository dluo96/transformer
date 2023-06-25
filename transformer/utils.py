import copy

from torch import nn


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers. They are identical in the sense that
    they share exactly the same architecture, but NOT parameters.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
