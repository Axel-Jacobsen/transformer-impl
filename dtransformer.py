#! /usr/bin/env python3

import torch

from torch import nn

from typing import Optional
from dataclasses import dataclass

from attention import MultiHeadAttention, generate_square_subsequent_mask


@dataclass
class EmbeddingParams:
    vocab_size: int
    max_sentence_size: int
    embedding_dimension_size: int


@dataclass
class MultiHeadAttentionParams:
    attention_dimension_size: int
    mid_dimension_size: int
    out_dimension_size: int  # just the embedding dim?
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
        self.embedder = nn.Embedding(
            embedding_params.vocab_size, embedding_params.embedding_dimension_size
        )
        self.layer_norm = [
            (
                nn.LayerNorm(embedding_params.embedding_dimension_size),
                nn.LayerNorm(embedding_params.embedding_dimension_size),
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(embedding_params.embedding_dimension_size)
        self.attention = [
            MultiHeadAttention(
                **attention_params.__dict__,
                **embedding_params.__dict__,
            )
            for _ in range(num_layers)
        ]
        self.mlp = [
            (
                nn.Linear(embedding_params.embedding_dimension_size, mlp_dim_size),
                nn.Linear(mlp_dim_size, embedding_params.embedding_dimension_size),
            )
            for _ in range(num_layers)
        ]
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, max_sentence_size)

        x is a batch of sequences of tokens
        returns a batch of sequences of embeddings

        ugly!
        """
        x = self.embedder(x)
        sentence_len, embedding_dim = x.shape
        causal_mask = generate_square_subsequent_mask(sentence_len, device=x.device)
        for n in range(self.num_layers):
            x = self.layer_norm[n][0](x)
            x = x + self.attention[n](x, x, mask=causal_mask)
            x = self.layer_norm[n][1](x)
            x = x + self.mlp[n][1](self.gelu(self.mlp[n][0](x)))

        x = self.final_layer_norm(x)
        if self.training:
            return nn.functional.softmax(x, dim=-1)
        return x
