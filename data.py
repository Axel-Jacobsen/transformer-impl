import torch

from torch.utils.data import Dataset

from pathlib import Path


class BasicTokenizer(Dataset):
    def __init__(self, raw_tokens, pad: bool = False) -> None:
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
        self._idx2token = {idx: token for token, idx in self._token2idx.items()}

    def tokenize(self, text: str) -> list[int]:
        tokens = [self._token2idx["<bos>"]]
        tokens += [self._token2idx[t] for t in text]
        tokens += [self._token2idx["<eos>"]]
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
        return "".join([self._idx2token[t] for t in tokens])


class PerLineDataset(Dataset):
    """Each line is a sequence of tokens, and assumed (poorly) to be iid"""

    def __init__(self, data_path: Path, pad: bool = False) -> None:
        with open(data_path, "r") as f:
            raw_text = f.read().lower()
            raw_tokens = sorted(list(set(raw_text)))

        self._pad = pad
        self._data = raw_text.lower().split("\n")
        self._max_len = max(len(line) for line in self._data) + 2

        self.tokenizer = BasicTokenizer(raw_tokens, pad=pad)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        full_seq = self.tokenizer.tokenize(self._data[index])
        input_seq = full_seq[:-1]
        output_seq = full_seq[1:]
        return torch.tensor(input_seq), torch.tensor(output_seq)

    @staticmethod
    def collate_fn(batch):
        xs, ys = zip(*batch)

        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True)

        return xs, ys

    @property
    def pad(self) -> bool:
        return self._pad

    @pad.setter
    def pad(self, pad: bool) -> None:
        self._pad = pad

    def get_tokens(self) -> list[str]:
        return self.tokenizer._tokens

    def vocab_size(self) -> int:
        return len(self.get_tokens())

    def max_size(self) -> int:
        return self._max_len
