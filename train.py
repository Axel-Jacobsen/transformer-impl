#! /usr/bin/env python3

import time

import torch

from torch import nn

from pathlib import Path

from tokenizer import NotGoodDatasetTokenizer
from dtransformer import DTransformer, EmbeddingParams, MultiHeadAttentionParams


def train():
    dset = NotGoodDatasetTokenizer(Path("canterbury_tales.txt"))

    device = torch.device("cpu")

    embedding_params = EmbeddingParams(
        vocab_size=dset.vocab_size(),
        max_sentence_size=dset.max_size(),
        embedding_dimension_size=256,
    )

    multi_head_attention_params = MultiHeadAttentionParams(
        attention_dimension_size=64,
        mid_dimension_size=256,
        out_dimension_size=embedding_params.embedding_dimension_size,
        num_heads=4,
    )

    transformer = DTransformer(
        num_layers=2,
        mlp_dim_size=256,
        embedding_params=embedding_params,
        attention_params=multi_head_attention_params,
    ).to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    global_step = 1
    t0 = time.perf_counter()
    for epoch in range(100):
        for x, y in dset:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            y_hat = transformer(x)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(
                    f"step: {global_step} epoch: {epoch} loss: {loss.item()} "
                    f"steps per second: {global_step / (time.perf_counter() - t0):.3f}"
                )

            global_step += 1


if __name__ == "__main__":
    train()
