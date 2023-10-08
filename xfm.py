#! /usr/bin/env python3

import torch


from torch import nn

from pathlib import Path

from tokenizer import Tokenizer


"""
working through formal algorithms for transformers

these will be very poor implementations. purpose is
for me to understand them as simply as possible.
"""


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

        # w_embedding transposed compared to "formal algorithms" since we have
        # to apply the function differently
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

    def basic_single_query_attention(
        self, token: torch.Tensor, context: list[torch.Tensor]
    ) -> torch.Tensor:
        """Very Very basic attention"""
        embedding = self.embed(token)
        embedded_context = [self.embed(t) for t in context]

        q = self._W_query(embedding)
        k = torch.stack([self._W_key(t) for t in embedded_context])
        v = torch.stack([self._W_value(t) for t in embedded_context])
        s = q @ k.mT / self._attention_dimension_size ** (1 / 2)

        alpha = torch.softmax(s, dim=-1)

        return torch.matmul(alpha, v)

    def attention(self, X, Z, mask=None):
        """Computes a single (masked) self- or cross- attention head

        (from Formal Algorithms)
        """
        embedding = self.embed(X)
        embedded_context = self.embed(Z)

        Q = self._W_query(embedding)
        K = self._W_key(embedded_context)
        V = self._W_value(embedded_context)
        S = K.mT @ Q / self._attention_dimension_size ** (1 / 2)

        if mask is not None:
            assert mask.shape == S.shape, f"{mask.shape=} {S.shape=}, should be same"
            S = S.masked_fill(mask, -1e9)
        return V @ torch.softmax(S, dim=-1)

    def forward(self, x, z, mask=None):
        return attention(x, z, mask=mask)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sentence_size: int,
        num_heads: int,
        mid_dim_size: int,
        embedding_dimension_size: int,
        attention_dimension_size: int,
        out_dimension_size: int,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._mid_dim_size = mid_dim_size
        self._vocab_size = vocab_size
        self._max_sentence_size = max_sentence_size
        self._embedding_dimension_size = embedding_dimension_size
        self._attention_dimension_size = attention_dimension_size
        self._out_dimension_size = out_dimension_size

        # w_embedding transposed compared to "formal algorithms" since we have
        # to apply the function differently
        self._W_embedding, self._W_positional = (
            nn.Linear(vocab_size, embedding_dimension_size, bias=False),
            nn.Linear(embedding_dimension_size, max_sentence_size, bias=False),
        )
        self._W_query = [
            nn.Linear(embedding_dimension_size, attention_dimension_size)
            for _ in range(num_heads)
        ]
        self._W_key = [
            nn.Linear(embedding_dimension_size, attention_dimension_size)
            for _ in range(num_heads)
        ]
        self._W_value = [
            nn.Linear(mid_dim_size, embedding_dimension_size) for _ in range(num_heads)
        ]
        # have to transpose this one too
        self._W0 = nn.Linear(num_heads * mid_dim_size, self._out_dimension_size)

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
    def _attention(self, embedding, embedded_context, Q, K, V, mask=None):
        q = Q(embedding)
        k = K(embedded_context)
        v = V(embedded_context)
        s = k.mT @ q / self._attention_dimension_size ** (1 / 2)

        if mask is not None:
            assert mask.shape == S.shape, f"{mask.shape=} {s.shape=}, should be same"
            s = s.masked_fill(mask, -1e9)

        return v @ torch.softmax(s, dim=-1)

    def multihead_attention(self, X, Z, mask=None):
        embedding = self.embed(X)
        embedded_context = self.embed(Z)

        print(f"{embedding.shape=}")
        outs = torch.cat(
            [
                self._attention(
                    embedding,
                    embedded_context,
                    self._W_query[h],
                    self._W_key[h],
                    self._W_value[h],
                    mask=mask,
                )
                for h in range(self._num_heads)
            ], dim=1
        )
        print(f"{self._W0=}")
        print(f"{outs.shape=}")

        return self._W0(outs)

    def forward(self, x, z, mask=None):
        return self.multihead_attention(x, z, mask=mask)


if __name__ == "__main__":
    data_path = Path("canterbury_tales.txt")
    tokenizer = Tokenizer(data_path, pad=True)
    attention = Attention(
        tokenizer.vocab_size(),
        tokenizer.max_size(),
        embedding_dimension_size=32,
        attention_dimension_size=32,
        out_dimension_size=32,
    )
    batch_size = 1

    print(f"vocab size: {tokenizer.vocab_size()}")

    batch = tokenizer[100 : 100 + batch_size]
    print(tokenizer._data[100])

    print(f"sentence shape is {batch.shape=}")

    ret = attention.token_embedding(batch)

    fin_basic = attention.basic_single_query_attention(
        batch[0, 11], list(batch[0, :11])
    )
    fin_normal = attention.attention(batch[0], batch[0])
    print(f"{fin_normal.shape=}")

    mh_attention = MultiHeadAttention(
        tokenizer.vocab_size(),
        tokenizer.max_size(),
        4,
        32,
        embedding_dimension_size=32,
        attention_dimension_size=32,
        out_dimension_size=32,
    )

    fin_mh = mh_attention(batch[0], batch[0])
    print(f"{fin_mh.shape=}")
