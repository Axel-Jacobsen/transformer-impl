#! /usr/bin/env python3

import torch

from torch import nn

from tqdm import tqdm

from typing import Optional
from dataclasses import dataclass

from embed import Embedding
from attention import MultiHeadAttention, generate_square_subsequent_mask


@dataclass
class EmbeddingParams:
    vocab_size: int
    max_sentence_size: int
    embedding_dimension_size: int


@dataclass
class MultiHeadAttentionParams:
    attention_dimension_size: int
    mid_dimension_size: int
    out_dimension_size: int  # just hte embedding dim???
    num_heads: Optional[int] = None


class DTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mlp_dim_size: int,
        embedding_params: EmbeddingParams,
        attention_params: MultiHeadAttentionParams,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim_size = mlp_dim_size
        self.embedder = Embedding(**embedding_params.__dict__)
        self.layer_norm = [
            (
                nn.LayerNorm(
                    [
                        embedding_params.max_sentence_size,
                        embedding_params.embedding_dimension_size,
                    ]
                ),
                nn.LayerNorm(
                    [
                        embedding_params.max_sentence_size,
                        embedding_params.embedding_dimension_size,
                    ]
                ),
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(
            [
                embedding_params.max_sentence_size,
                embedding_params.embedding_dimension_size,
            ]
        )
        self.attention = [
            MultiHeadAttention(
                **attention_params.__dict__,
                **embedding_params.__dict__,
            )
            for _ in range(num_layers)
        ]
        self.mlp = [
            (
                nn.Linear(embedding_params.embedding_dimension_size, mlp_dim_size),
                nn.Linear(mlp_dim_size, embedding_params.embedding_dimension_size),
            )
            for _ in range(num_layers)
        ]
        self.gelu = nn.GELU()
        self.causal_mask = generate_square_subsequent_mask(
            embedding_params.max_sentence_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, max_sentence_size)

        x is a batch of sequences of tokens
        returns a batch of sequences of embeddings

        ugly!
        """
        x = self.embedder(x)
        for n in range(self.num_layers):
            x = self.layer_norm[n][0](x)
            x = x + self.attention[n](x, x, mask=self.causal_mask)
            x = self.layer_norm[n][1](x)
            x = x + self.mlp[n][1](self.gelu(self.mlp[n][0](x)))

        x = self.final_layer_norm(x)
        x = self.embedder.unembed(x)
        return nn.functional.softmax(x, dim=-1)


if __name__ == "__main__":
    from itertools import cycle

    source_string = "aababbabac"
    seq = cycle(source_string)

    CONTEXT_WINDOW = 16
    VOCAB_SIZE = 4
    EMBEDDING_DIMENSION_SIZE = 3

    embedding_params = EmbeddingParams(
        VOCAB_SIZE, CONTEXT_WINDOW, EMBEDDING_DIMENSION_SIZE
    )
    attention_params = MultiHeadAttentionParams(
        attention_dimension_size=16,
        mid_dimension_size=16,
        out_dimension_size=EMBEDDING_DIMENSION_SIZE,
        num_heads=4,
    )

    transformer = DTransformer(
        num_layers=4,
        mlp_dim_size=8,
        embedding_params=embedding_params,
        attention_params=attention_params,
    )

    def tokenize(char: str) -> torch.Tensor:
        if char == "a":
            return torch.tensor(0)
        elif char == "b":
            return torch.tensor(1)
        elif char == "c":
            return torch.tensor(2)
        elif char == "<eos>":
            return torch.tensor(3)
        raise RuntimeError(f"can't tokenize {char=}")

    def detokenize(tensor: torch.Tensor) -> str:
        if tensor.argmax() == 0:
            return "a"
        elif tensor.argmax() == 1:
            return "b"
        elif tensor.argmax() == 2:
            return "c"
        elif tensor.argmax() == 3:
            return "<eos>"
        raise RuntimeError(f"can't detokenize {tensor=}")

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(10000))
    for i in pbar:
        optimizer.zero_grad()
        x = torch.stack([tokenize(next(seq)) for _ in range(CONTEXT_WINDOW)])
        y = torch.roll(x, shifts=-1, dims=0)
        y[-1] = tokenize(next(seq))

        y_hat = transformer(x)

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss = {loss.item()}")

    transformer.eval()

    raw_sentence = [next(seq) for _ in range(CONTEXT_WINDOW)]
    x = torch.stack([tokenize(c) for c in raw_sentence])
    y_hat = transformer(x)

    print("source =", source_string)
    print(f"input = {''.join(raw_sentence)}")
    print("output =", "".join([detokenize(t) for t in y_hat]))
