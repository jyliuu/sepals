"""Matplotlib visualization for fitted ``SeparatedALSRegressor`` models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .als import SeparatedALSRegressor


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install optional dependencies, e.g. "
            "`pip install sepals[plot]` or `uv add 'sepals[plot]'`."
        ) from e
    return plt


def plot_separation_stages(
    model: "SeparatedALSRegressor",
    *,
    n_grid: int = 200,
    figsize: Optional[Tuple[float, float]] = None,
) -> "Figure":
    """Plot rank-separated structure: one row per rank term, columns ``s_l`` then ``g_{l,k}``.

    Row ``l`` shows the scalar coefficient ``s_l`` and each component ``g_{l,k}``
    as a function of ``x_k`` on the training-scale domain ``[x_min_k, x_max_k]``.
    """
    check_is_fitted(model, attributes=["coeffs_", "scales_", "x_min_", "x_range_"])
    plt = _mpl()
    r = int(model.rank)
    p = model.n_features_in_
    if figsize is None:
        figsize = (2.2 * (p + 1), 1.8 * r)
    fig, axes = plt.subplots(r, p + 1, squeeze=False, figsize=figsize, layout="constrained")
    scales = np.asarray(model.scales_, dtype=float).ravel()
    for ell in range(r):
        ax_s = axes[ell, 0]
        ax_s.set_axis_off()
        l1 = ell + 1  # 1-based rank index l matching docstring notation
        ax_s.text(
            0.5,
            0.5,
            rf"$s_{{{l1}}} = {float(scales[ell]):.4g}$",
            transform=ax_s.transAxes,
            ha="center",
            va="center",
            fontsize="medium",
        )
        for k in range(p):
            ax = axes[ell, k + 1]
            lo = float(model.x_min_[k])
            hi = float(model.x_min_[k] + model.x_range_[k])
            grid = np.linspace(lo, hi, n_grid)
            g = model.factor_values(k, grid)
            ax.plot(grid, g[:, ell], color=f"C{(k % 9) + 1}")
            ax.axhline(0.0, color="0.5", linewidth=0.6, linestyle=":")
            ax.set_xlabel(rf"$x_{{{k + 1}}}$")
            if ell == 0:
                ax.set_title(rf"$g_{{{ell + 1},{k + 1}}}$")
    fig.suptitle("Separated ALS factors by rank term", fontsize="medium")
    return fig


def plot_single_stage(
    model: "SeparatedALSRegressor",
    stage: int,
    *,
    n_grid: int = 200,
    figsize: Optional[Tuple[float, float]] = None,
) -> "Figure":
    """Plot one rank term: ``s_l`` and all ``g_{l,k}`` for fixed ``stage`` (0-based index ``l``)."""
    check_is_fitted(model, attributes=["coeffs_", "scales_", "x_min_", "x_range_"])
    if stage < 0 or stage >= model.rank:
        raise ValueError(f"stage must be in [0, {model.rank}), got {stage}")
    plt = _mpl()
    p = model.n_features_in_
    if figsize is None:
        figsize = (2.2 * (p + 1), 2.0)
    fig, axes = plt.subplots(1, p + 1, squeeze=False, figsize=figsize, layout="constrained")
    row = axes[0]
    s = float(np.asarray(model.scales_, dtype=float).ravel()[stage])
    ax_s = row[0]
    ax_s.set_axis_off()
    l1 = stage + 1
    ax_s.text(
        0.5,
        0.5,
        rf"$s_{{{l1}}} = {s:.4g}$",
        transform=ax_s.transAxes,
        ha="center",
        va="center",
        fontsize="medium",
    )
    for k in range(p):
        ax = row[k + 1]
        lo = float(model.x_min_[k])
        hi = float(model.x_min_[k] + model.x_range_[k])
        grid = np.linspace(lo, hi, n_grid)
        g = model.factor_values(k, grid)
        ax.plot(grid, g[:, stage], color=f"C{(k % 9) + 1}")
        ax.axhline(0.0, color="0.5", linewidth=0.6, linestyle=":")
        ax.set_xlabel(rf"$x_{{{k + 1}}}$")
        ax.set_title(rf"$g_{{{stage + 1},{k + 1}}}$")
    fig.suptitle(rf"Rank term ${stage + 1}$", fontsize="medium")
    return fig
