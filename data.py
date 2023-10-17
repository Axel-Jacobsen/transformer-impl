import torch


from torch.utils.data import Dataset

from pathlib import Path


""" This is a dataset, not a tokenizer. Well, both.
"""


class NotGoodDatasetTokenizer(Dataset):
    def __init__(self, data_path: Path, pad: bool = False) -> None:
        with open(data_path, "r") as f:
            raw_text = f.read().lower()
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
        self._max_len = max(len(line) for line in self._data) + 2

    def __repr__(self) -> str:
        return f"Tokenizer(tokens={''.join(self._tokens)})"

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset
        """
        data = [self._token2idx["<bos>"]]
        data += [self._token2idx[t] for t in self._data[idx]]
        data += [self._token2idx["<eos>"]]

        input_seq = torch.tensor(data[:-1])
        target_seq = torch.tensor(data[1:])

        return input_seq, target_seq

    @property
    def pad(self) -> bool:
        return self._pad

    @pad.setter
    def pad(self, pad: bool) -> None:
        self._pad = pad

    def get_tokens(self) -> list[str]:
        return self._tokens

    def vocab_size(self) -> int:
        return len(self._tokens)

    def max_size(self) -> int:
        return self._max_len

    @staticmethod
    def collate_fn(batch):
        xs, ys = zip(*batch)

        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True)

        return xs, ys
