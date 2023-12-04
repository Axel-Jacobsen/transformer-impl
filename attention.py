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
        self._W_query = nn.ModuleList(
            [
                nn.Linear(embedding_dimension_size, attention_dimension_size)
                for _ in range(num_heads)
            ]
        )
        self._W_key = nn.ModuleList(
            [
                nn.Linear(embedding_dimension_size, attention_dimension_size)
                for _ in range(num_heads)
            ]
        )
        self._W_value = nn.ModuleList(
            [
                nn.Linear(embedding_dimension_size, mid_dimension_size)
                for _ in range(num_heads)
            ]
        )
        self._W0 = nn.Linear(num_heads * mid_dimension_size, out_dimension_size)

    def _attention(self, embedding, embedded_context, Q, K, V, mask=None):
        q = Q(embedding)
        k = K(embedded_context)
        v = V(embedded_context)
        s = q @ k.mT / self._attention_dimension_size ** (1 / 2)

        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))

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
                dim=-1,
            )
        )

    def forward(self, x, z, mask=None):
        return self.multihead_attention(x, z, mask=mask)
