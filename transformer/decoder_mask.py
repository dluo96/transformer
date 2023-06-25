import torch


def subsequent_mask(len_input: int) -> torch.Tensor:
    """Create a mask which masks subsequent positions. This is used
    during training. It is an attention mask which shows the position(s)
    that each target word (row) is allowed to look at (column). Words
    are blocked for attending to future words during training. In other
    words, this ensures that the predictions for position `i` can depend
    only on the known outputs at positions j <= i.

    Args:
        len_input: the length of the input sequence, i.e. the length of
            the current sequence of predicted words.

    Returns:
        Mask of shape (1, len_input, len_input) which is a boolean,
            lower triangular matrix.
    """
    # Create lower triangular matrix
    attn_shape = (1, len_input, len_input)
    array_ones = torch.ones(attn_shape)
    subseq_mask = torch.tril(array_ones).type(torch.bool)
    return subseq_mask
