from __future__ import annotations

import math
import numpy as np
import torch


def exact_energy(n: int, domain_length: float = 1.0) -> float:
    """Infinite-square-well eigenvalue for -psi'' = E psi on [0, L]."""
    return (n * math.pi / domain_length) ** 2


def exact_wavefunction_np(n: int, x: np.ndarray, domain_length: float = 1.0) -> np.ndarray:
    return np.sqrt(2.0 / domain_length) * np.sin(n * np.pi * x / domain_length)


def exact_wavefunction_torch(x: torch.Tensor, n: int, domain_length: float = 1.0) -> torch.Tensor:
    return math.sqrt(2.0 / domain_length) * torch.sin(n * math.pi * x / domain_length)


def relative_l2_np(y_num: np.ndarray, y_ref: np.ndarray, x: np.ndarray) -> float:
    num = np.sqrt(np.trapezoid((y_num - y_ref) ** 2, x))
    den = np.sqrt(np.trapezoid(y_ref ** 2, x))
    return float(num / den)
