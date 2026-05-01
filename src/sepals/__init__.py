"""Optimized separated ALS regression inspired by Beylkin et al."""

from .als import BasisKind, SeparatedALSRegressor
from .datasets import friedman1, friedman2, rmse

__all__ = [
    "BasisKind",
    "SeparatedALSRegressor",
    "friedman1",
    "friedman2",
    "rmse",
]

__version__ = "0.1.0"
