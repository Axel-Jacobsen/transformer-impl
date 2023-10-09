#! /usr/bin/env python3

import torch


from torch import nn

from pathlib import Path

from tokenizer import Tokenizer
from attention import MultiHeadAttention, Attention, Embedding


"""
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
"""


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
        for l in range(num_layers):
            pass


def basic_tests():
    data_path = Path("canterbury_tales.txt")
    tokenizer = Tokenizer(data_path, pad=True)

    embedder = Embedding(
        vocab_size=tokenizer.vocab_size(),
        max_sentence_size=tokenizer.max_size(),
        embedding_dimension_size=32,
    )

    attention = Attention(
        tokenizer.vocab_size(),
        tokenizer.max_size(),
        embedding_dimension_size=32,
        attention_dimension_size=32,
        out_dimension_size=32,
    )

    batch_size = 1
    batch = tokenizer[100 : 100 + batch_size]

    attention.basic_single_query_attention(
        embedder.embed(batch[0, 11]), [embedder.embed(b) for b in batch[0, :11]]
    )
    fin_normal = attention.attention(embedder.embed(batch[0]), embedder.embed(batch[0]))

    mh_attention = MultiHeadAttention(
        tokenizer.vocab_size(),
        tokenizer.max_size(),
        4,
        32,
        embedding_dimension_size=32,
        attention_dimension_size=32,
        out_dimension_size=32,
    )

    fin_mh = mh_attention(embedder.embed(batch[0]), embedder.embed(batch[0]))


if __name__ == "__main__":
    basic_tests()

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
