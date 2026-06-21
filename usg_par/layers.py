"""Reusable building blocks shared across USG-Par modules."""

from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple multi-layer perceptron with ReLU activations.

    Used for the subject/object projectors and mask-embedding heads.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else in_dim
        out_dim = out_dim if out_dim is not None else in_dim
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)