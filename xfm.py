#! /usr/bin/env python3

import torch


from torch import nn


# def basic_tests():
#     data_path = Path("canterbury_tales.txt")
#     tokenizer = Tokenizer(data_path, pad=True)
#     attention = Attention(
#         tokenizer.vocab_size(),
#         tokenizer.max_size(),
#         embedding_dimension_size=32,
#         attention_dimension_size=32,
#         out_dimension_size=32,
#     )
#     batch_size = 1

#     print(f"vocab size: {tokenizer.vocab_size()}")

#     batch = tokenizer[100 : 100 + batch_size]
#     print(tokenizer._data[100])

#     print(f"sentence shape is {batch.shape=}")

#     attention.token_embedding(batch)

#     attention.basic_single_query_attention(batch[0, 11], list(batch[0, :11]))
#     fin_normal = attention.attention(batch[0], batch[0])
#     print(f"{fin_normal.shape=}")

#     mh_attention = MultiHeadAttention(
#         tokenizer.vocab_size(),
#         tokenizer.max_size(),
#         4,
#         32,
#         embedding_dimension_size=32,
#         attention_dimension_size=32,
#         out_dimension_size=32,
#     )

#     fin_mh = mh_attention(batch[0], batch[0])
#     print(f"{fin_mh.shape=}")


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
