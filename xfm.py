#! /usr/bin/env python3

import torch


from torch import nn

from pathlib import Path

from tokenizer import Tokenizer


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sentence_size: int,
        embedding_dimension_size: int,
        attention_dimension_size: int,
        out_dimension_size: int,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._max_sentence_size = max_sentence_size
        self._embedding_dimension_size = embedding_dimension_size
        self._attention_dimension_size = attention_dimension_size
        self._out_dimension_size = out_dimension_size

        # transposed compared to "formal algorithms"
        self._W_embedding = nn.Linear(vocab_size, embedding_dimension_size, bias=False)
        self._W_positional = nn.Linear(
            embedding_dimension_size, max_sentence_size, bias=False
        )
        self._W_query = nn.Linear(embedding_dimension_size, attention_dimension_size)
        self._W_key = nn.Linear(embedding_dimension_size, attention_dimension_size)
        self._W_value = nn.Linear(out_dimension_size, embedding_dimension_size)

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

    def basic_attention(
        self, token: torch.Tensor, context: list[torch.Tensor]
    ) -> torch.Tensor:
        """Very Very basic attention"""
        embedding = self.embed(token)
        embedded_context = [self.embed(t) for t in context]

        q = self._W_query(embedding)
        k = torch.stack([self._W_key(t) for t in embedded_context])
        v = torch.stack([self._W_value(t) for t in embedded_context])

        alpha = torch.softmax(
            torch.matmul(q, k.mT) / self._attention_dimension_size ** (1 / 2), dim=-1
        )

        return torch.matmul(alpha, v)


if __name__ == "__main__":
    data_path = Path("canterbury_tales.txt")
    tokenizer = Tokenizer(data_path, pad=True)
    transformer = Transformer(
        tokenizer.vocab_size(),
        tokenizer.max_size(),
        embedding_dimension_size=128,
        attention_dimension_size=128,
        out_dimension_size=128,
    )
    batch_size = 1

    print(f"vocab size: {tokenizer.vocab_size()}")

    batch = tokenizer[100 : 100 + batch_size]
    print(tokenizer._data[100])

    print(f"sentence shape is {batch.shape=}")

    ret = transformer.token_embedding(batch)

    fin = transformer.basic_attention(batch[0, 11], list(batch[0, :11]))
    print(f"{fin.shape=}")
