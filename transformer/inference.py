"""Module for running inference with the transformer."""
import logging
import torch

from transformer.decoder_mask import subsequent_mask
from transformer.transformer_architecture import make_model


def inference_test() -> None:
    # Create a transformer with a vocabulary size of 11 words that has 2 attention heads
    test_model = make_model(src_vocab=11, tgt_vocab=11, N=2)

    # Set module in evaluation mode
    test_model.eval()

    # Create 10-word input sequence (list of indices each
    # corresponding to a word in the 11-word vocabulary)
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    # Pass input through encoder - output is used as keys and values in decoder
    memory = test_model.encode(src, src_mask)

    # Initialise container with one zero (start token) to be filled with predicted words
    ys = torch.zeros(1, 1).type_as(src)

    # Each iteration will predict a new word
    for _i in range(9):
        # Extract the length of the input i.e. the number of predicted words so far
        len_input = ys.size(1)

        # Create mask to ensure words are blocked from attending to future words
        tgt_mask = subsequent_mask(len_input).type_as(src.data)

        # Forward pass through decoder
        output_probs = test_model.decode(memory, src_mask, ys, tgt_mask)

        # Extract output probabilities for next word ONLY
        output_probs_next_word = output_probs[:, -1]

        # Generate the (log of) output probabilities
        # (one probability per word in the vocabulary)
        log_probs_next_word = test_model.generator(output_probs_next_word)

        # Select the next word (specifically its vocabulary index)
        # greedily based on the (log of) output probabilities
        _, next_word = torch.max(log_probs_next_word, dim=1)
        next_word = next_word.data[0]

        # Append the predicted word to the output sequence
        y = torch.empty(1, 1).type_as(src.data).fill_(next_word)
        ys = torch.cat([ys, y], dim=1)

    logging.info("Example Untrained Model Prediction:", ys)


def run_tests() -> None:
    for _ in range(10):
        inference_test()


if __name__ == "__main__":
    run_tests()
