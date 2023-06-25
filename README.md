# The Annotated Transformer - Annotated
The code in this (sub)package is based on [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) 
by Harvard NLP. I have further annotated the code including the following:
- Renaming objects that I found to be unclear or confusing. 
- Adding type hinting for static type checking.
- Writing detailed docstrings explaining methods and classes that I found unclear.
- Refactoring parts of the code to improve clarity.


Below is a figure showing which components of the original transformer architecture correspond to
which classes in this subpackage. 

![](../../img/annotated_transformer.png)

The transformer consists of the following classes and methods:
- [`Embeddings`](embedding.py) i.e. "Input Embedding" and "Output Embedding" in the figure.
- [`PositionalEncoding`](positional_encoding.py)
- [`EncoderLayer`](encoder.py)
- [`Encoder`](encoder.py)
- [`DecoderLayer`](decoder.py)
- [`Decoder`](decoder.py)
- [`LayerNorm`](layer_norm.py)
- [`SublayerConnection`](sublayer_connection.py)
- [`PositionwiseFeedForward`](feedforward_net.py) i.e. "Feed Forward" in the figure.
- [`MultiHeadedAttention`](attention.py) which includes "Masked Multi-Head Attention".
- [`Generator`](embedding.py)
- [`subsequent_mask`](decoder_mask.py)


## Setup

Clone the repository:

```shell
git clone git@github.com:dluo96/transformer.git
```

Create a conda environment:

```shell
conda env create --file environment.yaml
```

Activate the conda environment:

```shell
conda transformer-env
```

Install the local package in editable mode:

```shell
pip install --editable .
```

## References

