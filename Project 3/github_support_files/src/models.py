from __future__ import annotations

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class TrialNet(nn.Module):
    """Fully-connected network with hard homogeneous Dirichlet BCs on [0, L]."""

    def __init__(self, width: int = 32, depth: int = 2, activation: str = "tanh", domain_length: float = 1.0):
        super().__init__()
        if activation.lower() not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation {activation!r}. Choose from {sorted(_ACTIVATIONS)}")
        act = _ACTIVATIONS[activation.lower()]
        layers: list[nn.Module] = [nn.Linear(1, width), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), act()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.domain_length = float(domain_length)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.domain_length
        return x * (L - x) * self.net(x)
