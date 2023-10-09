#! /usr/bin/env python3

import torch


from torch import nn


class Embedding(nn.Module):
    def __init__(
        self, vocab_size: int, max_sentence_size: int, embedding_dimension_size: int
    ):
        super().__init__()
        self._vocab_size = vocab_size
        self._max_sentence_size = max_sentence_size
        self._embedding_dimension_size = embedding_dimension_size

        self._W_embedding, self._W_positional = (
            nn.Linear(vocab_size, embedding_dimension_size, bias=False),
            nn.Linear(embedding_dimension_size, max_sentence_size, bias=False),
        )

    def token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embedding of a sequence of tokens

        Tokens should be a tensor of shape (batch_size, sequence_length)
        Returns a shape of
            (batch_size, sequence_length, embedding_dimension_size)
        """
        return self._W_embedding(
            nn.functional.one_hot(tokens, self._vocab_size).float()
        )

    def positional_embedding(self) -> torch.Tensor:
        """Positional embedding of a sequence of tokens"""
        return self._W_positional.weight

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed a sequence of tokens"""
        return self.token_embedding(tokens) + self.positional_embedding()

    def unembed(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        """Unembed a sequence of tokens

        We can make this it's own learned layer, but you can also
        just transpose the embedding matrix and multiply it by the
        embedded tokens. Easier, good 'nough for now
        """
        return embedded_tokens @ self._W_embedding.weight.mT
