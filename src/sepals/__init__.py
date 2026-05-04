"""Optimized separated ALS regression inspired by Beylkin et al."""

from .als import BasisKind, KernelBackend, SeparatedALSRegressor
from .datasets import friedman1, friedman2, rmse

__all__ = [
    "BasisKind",
    "KernelBackend",
    "SeparatedALSRegressor",
    "friedman1",
    "friedman2",
    "rmse",
]

__version__ = "0.1.0"
