"""Optimized separated ALS regression inspired by Beylkin et al."""

from .als import BasisKind, KernelBackend, SeparatedALSRegressor
from .datasets import friedman1, friedman2, rmse
from .plotting import plot_separation_stages, plot_single_stage

__all__ = [
    "BasisKind",
    "KernelBackend",
    "SeparatedALSRegressor",
    "friedman1",
    "friedman2",
    "rmse",
    "plot_separation_stages",
    "plot_single_stage",
]

__version__ = "0.1.0"
