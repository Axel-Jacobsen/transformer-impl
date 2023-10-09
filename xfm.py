#! /usr/bin/env python3

import torch

from torch import nn

from typing import Optional
from dataclasses import dataclass

from embed import Embedding
from layer_norm import LayerNorm
from attention import MultiHeadAttention, generate_square_subsequent_mask

torch.autograd.set_detect_anomaly(True)


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
                LayerNorm(embedding_params.embedding_dimension_size),
                LayerNorm(embedding_params.embedding_dimension_size),
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = LayerNorm(embedding_params.embedding_dimension_size)
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
        for l in range(self.num_layers):
            print(f"{x=}")
            print("a", x.shape)
            x = self.layer_norm[l][0](x)
            print("b", x.shape)
            x += self.attention[l](x, x, mask=self.causal_mask)
            print("c", x.shape)
            x = self.layer_norm[l][1](x)
            print("d", x.shape)
            x += self.mlp[l][1](self.gelu(self.mlp[l][0](x.T))).T
            print("e", x.shape)

        print(x)
        x = self.final_layer_norm(x)
        print(f"{x.shape=}")
        x = self.embedder.unembed(x)
        return nn.functional.softmax(x, dim=-1)


if __name__ == "__main__":
    from itertools import cycle

    seq = cycle("aab")

    CONTEXT_WINDOW = 10
    VOCAB_SIZE = 3

    def tokenize(char: str) -> torch.Tensor:
        if char == "a":
            return torch.tensor(0)
        elif char == "b":
            return torch.tensor(1)
        elif char == "<eos>":
            return torch.tensor(2)
        raise RuntimeError(f"can't tokenize {char=}")

    def detokenize(tensor: torch.Tensor) -> str:
        if tensor.argmax() == 0:
            return "a"
        elif tensor.argmax() == 1:
            return "b"
        elif tensor.argmax() == 2:
            return "<eos>"
        raise RuntimeError(f"can't detokenize {tensor=}")

    embedding_params = EmbeddingParams(VOCAB_SIZE, CONTEXT_WINDOW, 3)
    attention_params = MultiHeadAttentionParams(
        attention_dimension_size=4,
        mid_dimension_size=5,
        out_dimension_size=3,
        num_heads=2,
    )

    transformer = DTransformer(
        num_layers=2,
        mlp_dim_size=4,
        embedding_params=embedding_params,
        attention_params=attention_params,
    )

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for i in range(100):
        optimizer.zero_grad()
        x = torch.stack([tokenize(next(seq)) for _ in range(CONTEXT_WINDOW)])
        y = torch.clone(x)

        print(x.shape)
        y_hat = transformer(x)
        print(y_hat, y)
        loss = criterion(y_hat.T, y)
        loss.backward()
        optimizer.step()
        print(f"{i=} {loss.item()=}")

    x = torch.stack([tokenize(next(seq)) for _ in range(CONTEXT_WINDOW)])
    transformer.eval()
    y_hat = transformer(x)
    print("".join([detokenize(y) for y in y_hat.argmax(dim=-1)]))
