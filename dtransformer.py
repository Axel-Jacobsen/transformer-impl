import torch

from torch import nn

from typing import Optional
from dataclasses import dataclass

from attention import MultiHeadAttention


@dataclass
class EmbeddingParams:
    vocab_size: int
    max_sentence_size: int
    embedding_dimension_size: int


@dataclass
class MultiHeadAttentionParams:
    attention_dimension_size: int
    mid_dimension_size: int
    out_dimension_size: int  # just the embedding dim?
    num_heads: Optional[int] = None


def generate_square_subsequent_mask(
    sz: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
    return (
        mask.float().masked_fill(mask == 0, 0.0).masked_fill(mask == 1, float("-inf"))
    )


class DTransformerLayer(nn.Module):
    def __init__(
        self,
        mlp_dim_size: int,
        embedding_params: EmbeddingParams,
        attention_params: MultiHeadAttentionParams,
    ):
        super().__init__()
        self.mlp_dim_size = mlp_dim_size

        self.layer_norm_1 = nn.LayerNorm(embedding_params.embedding_dimension_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_params.embedding_dimension_size)
        self.layer_norm_3 = nn.LayerNorm(embedding_params.embedding_dimension_size)
        self.attention = MultiHeadAttention(
            **attention_params.__dict__,
            **embedding_params.__dict__,
        )
        self.mlp_1 = nn.Linear(embedding_params.embedding_dimension_size, mlp_dim_size)
        self.mlp_2 = nn.Linear(mlp_dim_size, embedding_params.embedding_dimension_size)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, max_sentence_size)

        x is a batch of sequences of tokens
        returns a batch of sequences of embeddings
        """
        _, sentence_len, _ = x.shape
        causal_mask = generate_square_subsequent_mask(sentence_len, device=x.device)

        x = self.layer_norm_1(x)
        x = x + self.attention(x, x, mask=causal_mask)
        x = self.layer_norm_2(x)
        x = x + self.mlp_2(self.gelu(self.mlp_1(x)))
        x = self.layer_norm_3(x)

        return x


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
        self.embedder = nn.Embedding(
            embedding_params.vocab_size, embedding_params.embedding_dimension_size
        )
        self.model = nn.Sequential(
            *[
                DTransformerLayer(mlp_dim_size, embedding_params, attention_params)
                for _ in range(num_layers)
            ]
        )
        self.final_linear = nn.Linear(
            embedding_params.embedding_dimension_size, embedding_params.vocab_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, max_sentence_size)

        x is a batch of sequences of tokens
        returns a batch of sequences of embeddings
        """
        x = self.embedder(x)
        x = self.model(x)
        x = self.final_linear(x)

        if not self.training:
            return nn.functional.softmax(x, dim=-1)
        return x
