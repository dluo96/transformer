import copy

import torch
from torch import nn

from transformer import (
    Decoder,
    Embeddings,
    Encoder,
    Generator,
    MultiHeadedAttention,
    PositionwiseFeedForward,
)
from transformer.decoder import DecoderLayer
from transformer.encoder import EncoderLayer
from transformer.positional_encoding import PositionalEncoding


class EncoderDecoder(nn.Module):
    """A standard encoder-decoder architecture.

    A few remarks:
        - `src_embed` consists of an embedding lookup table followed by positional
            encoding. Is used to preprocess the input that will be passed to the
            encoder.
        - `tgt_embed` consists of an embedding lookup table followed by positional
            encoding. Is used to preprocess the input that will be passed to the
            decoder.
        - `generator` is used to compute the output probabilities for each word in the
            vocabulary. These probabilities can be used to extract the next word. Thus,
            it is used to postprocess the output of the decoder.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src: torch.LongTensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Take in and process masked source and target sequences. The output of the
        encoder is passed to the decoder.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.LongTensor, src_mask: torch.Tensor) -> torch.Tensor:
        """First, get the input embedding (from the lookup table) and add the positional
        encoding. Then, pass the resultant embedding and the source mask into the
        encoder.

        Args:
            src: input to the transformer on the left side.
            src_mask: ...

        Returns:
            Output of encoder.
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """First, get the output embedding (from the lookup table) and add the
        positional encoding. Then, pass the resultant embedding as well as the
        memory, source mask, and target mask into the decoder.

        Returns:
            Output of decoder.
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """Construct a model (representing the transformer) from hyperparameters.
    Importantly, the `Embedding` module and `PositionalEncoding` modules are
    combined into a `Sequential` module.

    Args:
        src_vocab: size of the source vocabulary.
        tgt_vocab: size of the target vocabulary.
        N: the number of encoder stacks (or equivalently the number of decoder stacks).
        d_model: the embedding dimensionality.
        d_ff: the dimensionality of the projection in the position wise FFN.
        h: the number of heads in the multihead attention.
        dropout: the probability of an element being zeroed.

    Returns:
        The model representing the transformer architecture.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == "__main__":
    e = Embeddings(512, 10000)
    y = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    z = e(y)
    assert z[0, 2] == z[1, 0]
