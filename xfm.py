#! /usr/bin/env python3

import torch


from torch import nn

from pathlib import Path

from shitty_tokenizer import Tokenizer


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sentence_size: int,
        embedding_dimension_size: int,
        attention_dimension_size: int,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._max_sentence_size = max_sentence_size
        self._embedding_dimension_size = embedding_dimension_size
        self._attention_dimension_size = attention_dimension_size

        # transposed compared to "formal algorithms"
        self._W_embedding = nn.Linear(vocab_size, embedding_dimension_size, bias=False)
        self._W_positional = nn.Linear(
            embedding_dimension_size, max_sentence_size, bias=False
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
        """
        Embed a sequence of tokens
        """
        return self.token_embedding(tokens) + self.positional_embedding()

    def basic_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(tokens)

        return embedding


if __name__ == "__main__":
    data_path = Path("canterbury_tales.txt")
    tokenizer = Tokenizer(data_path)
    transformer = Transformer(
        tokenizer.vocab_size(),
        tokenizer.max_size(),
        embedding_dimension_size=128,
        attention_dimension_size=128,
    )
    batch_size = 16
    print(f"vocab size: {tokenizer.vocab_size()}")
    batch = tokenizer[100 : 100 + batch_size]
    print(f"sentence shape is {batch.shape=}")
    ret = transformer.token_embedding(batch)
    print(f"token embedding {ret.shape=}")
    print(f"{transformer.embed(batch).shape=}")
