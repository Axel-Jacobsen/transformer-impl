#! /usr/bin/env python3

import torch

from torch import nn

from typing import Optional
from dataclasses import dataclass

from embed import Embedding
from layer_norm import LayerNorm
from attention import MultiHeadAttention, generate_square_subsequent_mask


@dataclass
class EmbeddingParams:
    vocab_size: int
    max_sentence_size: int
    embedding_dim_size: int


@dataclass
class MultiHeadAttentionParams:
    embedding_params: EmbeddingParams
    attention_dim_size: int
    mid_dim_size: int
    out_dim_size: int
    num_heads: Optional[int] = None


class DTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mlp_dim_size: int,
        embedding_params: EmbeddingParams,
        attention_params: MultiHeadAttentionParams,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim_size = mlp_dim_size
        self.embedder = Embedding(**embedding_params.__dict__)
        self.layer_norm = [
            (
                LayerNorm(embedding_params.embedding_dim_size),
                LayerNorm(embedding_params.embedding_dim_size),
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = LayerNorm(embedding_params.embedding_dim_size)
        self.attention = [
            MultiHeadAttention(**attention_params.__dict__) for _ in range(num_layers)
        ]
        self.mlp = [
            (
                nn.Linear(embedding_params.embedding_dim_size, mlp_dim_size),
                nn.Linear(mlp_dim_size, embedding_params.embedding_dim_size),
            )
            for _ in range(num_layers)
        ]
        self.gelu = nn.GELU()
        self.causal_mask = generate_square_subsequent_mask(
            embedding_params.max_sentence_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, max_sentence_size)

        x is a batch of sequences of tokens
        returns a batch of sequences of embeddings

        ugly!
        """
        x = self.embedder(x)
        for l in range(self.num_layers):
            x = self.layer_norm[l][0](x)
            x += self.attention[l](x, x, mask=self.causal_mask)
            x = self.layer_norm[l][1](x)
            x += self.mlp[l][1](self.gelu(self.mlp[l][0](x)))

        x = self.final_layer_norm(x)
        x = self.embedder.unembed(x)
        return nn.functional.softmax(x, dim=-1)


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
