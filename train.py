#! /usr/bin/env python3

import torch

from torch import nn

from pathlib import Path
from typing import Optional

from embed import Embedding
from tokenizer import Tokenizer
from dtransformer import DTransformer, EmbeddingParams, MultiHeadAttentionParams


if __name__ == "__main__":
    t = Tokenizer(Path("canterbury_tales.txt"))

    embedding_params = EmbeddingParams(
        vocab_size=t.vocab_size(),
        max_sentence_size=t.max_size(),
        embedding_dimension_size=256,
    )

    multi_head_attention_params = MultiHeadAttentionParams(
        attention_dimension_size=64,
        mid_dimension_size=256,
        out_dimension_size=embedding_params.embedding_dimension_size,
        num_heads=4,
    )

    transformer = DTransformer(
        num_layers = 2,
        mlp_dim_size = 256,
        embedding_params=embedding_params,
        attention_params=multi_head_attention_params,
    )

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(100):

        for (x, y) in t:
            global_step += 1

            optimizer.zero_grad()

            y_hat = transformer(x)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(f"step: {global_step}, Loss: {loss.item()}")
