#! /usr/bin/env python3

import torch


from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dimension_size: int) -> None:
        super().__init__()
        self._dimension_size = dimension_size
        self._gamma = nn.Parameter(torch.ones(dimension_size))
        self._beta = nn.Parameter(torch.zeros(dimension_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.mean(x, dim=-1, keepdim=True)
        s = torch.std(x, dim=-1, keepdim=True)
        return (x - m) / s * self._gamma + self._beta
