#! /usr/bin/env python3

import time

import torch

from torch import nn
from torch.utils.data import DataLoader

from pathlib import Path

from data import NotGoodDatasetTokenizer
from dtransformer import DTransformer, EmbeddingParams, MultiHeadAttentionParams


def give_me_a_dataloader():
    dset = NotGoodDatasetTokenizer(Path("canterbury_tales.txt"))
    return DataLoader(dset, batch_size=32, shuffle=True, collate_fn=dset.collate_fn)


def train():
    train_dloader = give_me_a_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_params = EmbeddingParams(
        vocab_size=train_dloader.dataset.vocab_size(),
        max_sentence_size=train_dloader.dataset.max_size(),
        embedding_dimension_size=64,
    )

    multi_head_attention_params = MultiHeadAttentionParams(
        attention_dimension_size=64,
        mid_dimension_size=64,
        out_dimension_size=embedding_params.embedding_dimension_size,
        num_heads=4,
    )

    transformer = DTransformer(
        num_layers=4,
        mlp_dim_size=64,
        embedding_params=embedding_params,
        attention_params=multi_head_attention_params,
    ).to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss().to(device)

    global_step = 1
    t0 = time.perf_counter()
    for epoch in range(100):
        for x, y in train_dloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            y_hat = transformer(x)
            loss = criterion(y_hat.permute((0, 2, 1)), y)

            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(
                    f"step: {global_step} epoch: {epoch} loss: {loss.item()} "
                    f"steps per second: {global_step / (time.perf_counter() - t0):.3f}"
                )

            global_step += 1
    torch.save(transformer.state_dict(), "transformer.pth")


if __name__ == "__main__":
    train()
