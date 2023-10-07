#! /usr/bin/env python3

import torch

from torch import nn
from torch.utils.data import Dataset

from pathlib import Path


class Tokenizer(Dataset):
    def __init__(self, data_path: Path) -> None:
        with open(data_path, "r") as f:
            raw_text = f.read()
            raw_tokens = sorted(list(set(raw_text)))

        self._tokens = raw_tokens + ["<mask>", "<bos>", "<eos>", "<pad>"]
        self._token2idx = {token: idx for idx, token in enumerate(self._tokens)}
        self._data = raw_text.split("\n")

    def __repr__(self) -> str:
        return f"Tokenizer(tokens={''.join(self._tokens)})"

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([self._token2idx[t] for t in self._data[idx]])

    def get_tokens(self) -> list[str]:
        return self._tokens

    def vocab_size(self) -> int:
        return len(self._tokens)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dimension_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._W_embedding = nn.Linear(
            vocab_size,
            embedding_dimension_size,
            bias=False
        )

    def token_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Tokens should be a tensor of shape (batch_size, sequence_length)
        """
        return self._W_embedding(
            nn.functional.one_hot(tokens, self._vocab_size).float()
        )

    def get_embedding(self, token: int) -> torch.Tensor:
        return self._W_embedding.weight[:, token]

    def positional_embedding(self):
        pass


if __name__ == "__main__":
    data_path = Path("canterbury_tales.txt")
    tokenizer = Tokenizer(data_path)
    transformer = Transformer(tokenizer.vocab_size(), 128)
    print(f"vocab size: {tokenizer.vocab_size()}")

    sentence = tokenizer[100]
    print(tokenizer._data[100])
    print(sentence)
    ret = transformer.token_embedding(sentence)
    print(f"token embedding {ret.shape=}")
    print(ret[0])
    print(transformer.get_embedding(sentence[0]))
