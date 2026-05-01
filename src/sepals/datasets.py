from __future__ import annotations

from typing import Tuple

import numpy as np


def friedman1(
    n: int,
    rng: np.random.Generator,
    noise_std: float = 0.0,
    p: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Friedman 1 synthetic regression data."""
    X = rng.uniform(0.0, 1.0, size=(n, p))
    y = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )
    if noise_std > 0:
        y = y + rng.normal(0.0, noise_std, size=n)
    return X, y


def friedman2(
    n: int,
    rng: np.random.Generator,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Friedman 2 synthetic regression data."""
    X = np.empty((n, 4), dtype=float)
    X[:, 0] = rng.uniform(0.0, 100.0, size=n)
    X[:, 1] = rng.uniform(40.0 * np.pi, 560.0 * np.pi, size=n)
    X[:, 2] = rng.uniform(0.0, 1.0, size=n)
    X[:, 3] = rng.uniform(1.0, 11.0, size=n)
    y = np.sqrt(
        X[:, 0] ** 2
        + (X[:, 1] * X[:, 2] - 1.0 / (X[:, 1] * X[:, 3])) ** 2
    )
    if noise_std > 0:
        y = y + rng.normal(0.0, noise_std, size=n)
    return X, y


def rmse(y, yhat) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))
