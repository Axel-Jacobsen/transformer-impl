import torch


from torch.utils.data import Dataset

from typing import Union
from pathlib import Path


class Tokenizer(Dataset):
    def __init__(self, data_path: Path, pad: bool = False) -> None:
        with open(data_path, "r") as f:
            raw_text = f.read()
            raw_tokens = sorted(list(set(raw_text)))

        self._pad = pad
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
                [
                    self._get_item(i, pad=self.pad)
                    for i in range(*idx.indices(len(self)))
                ]
            )
        return self._get_item(idx)

    @property
    def pad(self) -> bool:
        return self._pad

    @pad.setter
    def pad(self, pad: bool) -> None:
        self._pad = pad

    def _get_item(self, idx: int, pad: bool = False) -> torch.Tensor:
        """
        Get a single item from the dataset

        Returns a tensor of shape (max_len + 1), with padding at the
        end of the actual data, and an <eos> token at the end of the
        sequence.
        """
        data = [self._token2idx[t] for t in self._data[idx]]
        data += [self._token2idx["<eos>"]]
        if pad:
            data += [self._token2idx["<pad>"]] * (self._max_len - len(self._data[idx]))
        return torch.tensor(data)

    def get_tokens(self) -> list[str]:
        return self._tokens

    def vocab_size(self) -> int:
        return len(self._tokens)

    def max_size(self) -> int:
        return self._max_len + 1
