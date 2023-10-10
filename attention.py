#! /usr/bin/env python3

import torch

from torch import nn

from typing import Optional
from dataclasses import dataclass


@dataclass
class EmbeddingParams:
    vocab_size: int
    max_sentence_size: int
    embedding_dim_size: int


@dataclass
class AttentionParams:
    attention_dimension_size: int
    mid_dimension_size: int
    out_dim_size: int
    num_heads: Optional[int] = None


def generate_square_subsequent_mask(
    sz: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    """Generate a square causal mask for the sequence.
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).

    directly from
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
    """
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


class Attention(nn.Module):
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

        self._W_query = nn.Linear(embedding_dimension_size, attention_dimension_size)
        self._W_key = nn.Linear(embedding_dimension_size, attention_dimension_size)
        self._W_value = nn.Linear(out_dimension_size, embedding_dimension_size)

    def basic_single_query_attention(
        self, embedding: torch.Tensor, embedded_context: list[torch.Tensor]
    ) -> torch.Tensor:
        """very very basic attention"""
        q = self._W_query(embedding)
        k = torch.stack([self._W_key(t) for t in embedded_context])
        v = torch.stack([self._W_value(t) for t in embedded_context])
        s = q @ k.mT / self._attention_dimension_size ** (1 / 2)

        alpha = torch.softmax(s, dim=-1)

        return torch.matmul(alpha, v)

    def attention(self, embedding, embedded_context, mask=None):
        """Computes a single (masked) self- or cross- attention head

        (from Formal Algorithms)
        """
        Q = self._W_query(embedding)
        K = self._W_key(embedded_context)
        V = self._W_value(embedded_context)
        S = K.mT @ Q / self._attention_dimension_size ** (1 / 2)

        if mask is not None:
            assert mask.shape == S.shape, f"{mask.shape=} {S.shape=}, should be same"
            S = S.masked_fill(mask, -1e9)
        return V @ torch.softmax(S, dim=-1)

    def forward(self, x, z, mask=None):
        return self.attention(x, z, mask=mask)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sentence_size: int,
        num_heads: int,
        mid_dimension_size: int,
        embedding_dimension_size: int,
        attention_dimension_size: int,
        out_dimension_size: int,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._mid_dimension_size = mid_dimension_size
        self._vocab_size = vocab_size
        self._max_sentence_size = max_sentence_size
        self._embedding_dimension_size = embedding_dimension_size
        self._attention_dimension_size = attention_dimension_size
        self._out_dimension_size = out_dimension_size

        # w_embedding transposed compared to "formal algorithms" since we have
        # to apply the function differently
        self._W_query = [
            nn.Linear(embedding_dimension_size, attention_dimension_size)
            for _ in range(num_heads)
        ]
        self._W_key = [
            nn.Linear(embedding_dimension_size, attention_dimension_size)
            for _ in range(num_heads)
        ]
        self._W_value = [
            nn.Linear(embedding_dimension_size, mid_dimension_size)
            for _ in range(num_heads)
        ]
        # have to transpose this one too
        self._W0 = nn.Linear(num_heads * mid_dimension_size, self._out_dimension_size)

    def _attention(self, embedding, embedded_context, Q, K, V, mask=None):
        q = Q(embedding)
        k = K(embedded_context)
        v = V(embedded_context)
        s = q @ k.T / self._attention_dimension_size ** (1 / 2)

        if mask is not None:
            assert mask.shape == s.shape, f"{mask.shape=} {s.shape=}, should be same"
            s *= mask

        return torch.softmax(s, dim=-1) @ v

    def multihead_attention(self, X, Z, mask=None):
        return self._W0(
            torch.cat(
                [
                    self._attention(
                        X,
                        Z,
                        self._W_query[h],
                        self._W_key[h],
                        self._W_value[h],
                        mask=mask,
                    )
                    for h in range(self._num_heads)
                ],
                dim=1,
            )
        )

    def forward(self, x, z, mask=None):
        return self.multihead_attention(x, z, mask=mask)
