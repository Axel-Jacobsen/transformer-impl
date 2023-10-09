#! /usr/bin/env python3

import torch

from torch import nn

from embed import Embedding
from layer_norm import LayerNorm
from attention import MultiHeadAttention


class DTransformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        mid_dim_size: int,
        vocab_size: int,
        max_sentence_size: int,
        embedding_dim_size: int,
        attention_dim_size: int,
        out_dim_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.embedder = Embedding(vocab_size, max_sentence_size, embedding_dim_size)
        self.attention_blocks = (
            MultiHeadAttention(
                vocab_size,
                max_sentence_size,
                num_heads,
                mid_dim_size,
                embedding_dim_size,
                attention_dim_size,
                out_dim_size,
            )
            for _ in range(num_layers)
        )
        self.layer_norms = (LayerNorm(embedding_dim_size) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, max_sentence_size)

        x is a batch of sequences of tokens
        """
        x = self.embedder(x)
        for attention_block, layer_norm in zip(self.attention_blocks, self.layer_norms):
            x = attention_block(x)
            x = layer_norm(x)
        return x


if __name__ == "__main__":
    from itertools import cycle

    seq = cycle("aab")

    CONTEXT_WINDOW = 5
    VOCAB_SIZE = 3

    def tokenize(char: str) -> torch.Tensor:
        if char == "a":
            return nn.functional.one_hot(torch.tensor(0), 3)
        elif char == "b":
            return nn.functional.one_hot(torch.tensor(1), 3)
        elif char == "<eos>":
            return nn.functional.one_hot(torch.tensor(2), 3)
        raise RuntimeError(f"can't tokenize {char=}")
