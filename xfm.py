#! /usr/bin/env python3

import torch


from torch import nn
from torch.utils.data import Dataset

from typing import Union
from pathlib import Path


class Tokenizer(Dataset):
    def __init__(self, data_path: Path) -> None:
        with open(data_path, "r") as f:
            raw_text = f.read()
            raw_tokens = sorted(list(set(raw_text)))

        self._tokens = raw_tokens + ["<mask>", "<bos>", "<eos>", "<pad>"]
        self._token2idx = {token: idx for idx, token in enumerate(raw_tokens)}
        self._token2idx.update(
            {
                "<mask>": len(raw_tokens),
                "<bos>": len(raw_tokens) + 1,
                "<eos>": len(raw_tokens) + 2,
                "<pad>": len(raw_tokens) + 3,
            }
        )
        self._data = raw_text.split("\n")
        self._max_len = max(len(line) for line in self._data)

    def __repr__(self) -> str:
        return f"Tokenizer(tokens={''.join(self._tokens)})"

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        if isinstance(idx, slice):
            return torch.stack(
                [self._get_item(i) for i in range(*idx.indices(len(self)))]
            )
        return self._get_item(idx)

    def _get_item(self, idx: int) -> torch.Tensor:
        """
        Get a single item from the dataset

        Returns a tensor of shape (max_len + 1), with padding at the
        end of the actual data, and an <eos> token at the end of the
        sequence.
        """

        return torch.tensor(
            [self._token2idx[t] for t in self._data[idx]]
            + [self._token2idx["<pad>"]] * (self._max_len - len(self._data[idx]))
            + [self._token2idx["<eos>"]]
        )

    def get_tokens(self) -> list[str]:
        return self._tokens

    def vocab_size(self) -> int:
        return len(self._tokens)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dimension_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._W_embedding = nn.Linear(vocab_size, embedding_dimension_size, bias=False)

    def token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Token embedding of a sequence of tokens

        Tokens should be a tensor of shape (batch_size, sequence_length)
        Returns a shape of (batch_size, sequence_length, embedding_dimension_size)
        """
        return self._W_embedding(
            nn.functional.one_hot(tokens, self._vocab_size).float()
        )

    def get_embedding(self, token: int) -> torch.Tensor:
        """
        Get
        """
        return self._W_embedding.weight[:, token]

    def positional_embedding(self):
        pass


if __name__ == "__main__":
    data_path = Path("canterbury_tales.txt")
    tokenizer = Tokenizer(data_path)
    transformer = Transformer(tokenizer.vocab_size(), 128)
    print(f"vocab size: {tokenizer.vocab_size()}")

    batch = tokenizer[100:110]
    print(f"{batch.shape=}")
    ret = transformer.token_embedding(batch)
    print(f"token embedding {ret.shape=}")
