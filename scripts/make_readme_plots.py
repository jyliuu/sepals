"""Generate README example figures (Friedman 1/2, California housing).

Run from repo root::

    uv run --extra plot python scripts/make_readme_plots.py

Requires matplotlib (``sepals[plot]``).
"""

from __future__ import annotations

import argparse
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.datasets import fetch_california_housing, make_friedman1

from sepals import SeparatedALSRegressor, friedman1, friedman2, plot_separation_stages

DEFAULT_REL_IMAGES = Path("docs") / "images"
SCRIPT_DIR = Path(__file__).resolve().parent


def _save_separation(model: SeparatedALSRegressor, path: Path, *, n_grid: int) -> None:
    fig = plot_separation_stages(model, n_grid=n_grid)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def example1_sklearn_friedman1(out_dir: Path) -> None:
    X, y = make_friedman1(n_samples=600, n_features=5, noise=0.1, random_state=0)
    model = SeparatedALSRegressor(
        rank=3,
        degree=3,
        max_sweeps=20,
        random_state=0,
        verbose=False,
    )
    model.fit(X, y)
    _save_separation(model, out_dir / "friedman1_sklearn_separation_stages.png", n_grid=150)


def example2_package_friedman1_p10(out_dir: Path) -> None:
    rng = np.random.default_rng(123)
    X_train, y_train = friedman1(2_000, rng, p=10)
    model = SeparatedALSRegressor(
        rank=4,
        degree=5,
        basis="monomial",
        max_sweeps=40,
        n_init=2,
        random_state=123,
        fit_intercept=True,
    )
    model.fit(X_train, y_train)
    _save_separation(model, out_dir / "friedman1_p10_separation_stages.png", n_grid=120)


def example3_friedman2_tent(out_dir: Path) -> None:
    rng = np.random.default_rng(123)
    X, y = friedman2(5_000, rng)
    model = SeparatedALSRegressor(
        rank=4,
        degree=6,
        basis="tent",
        smoothness=1e-6,
        penalty_kind="tent_level",
        max_sweeps=20,
        random_state=123,
    )
    model.fit(X, y)
    _save_separation(model, out_dir / "friedman2_tent_separation_stages.png", n_grid=120)


def example4_california_housing(out_dir: Path) -> None:
    # Hyperparameters from MPF2 interpretable search (sepals_rank_le_2_ctr23, california_housing).
    data = fetch_california_housing()
    X_full = np.asarray(data.data, dtype=float)
    y_full = np.asarray(data.target, dtype=float).ravel()
    # Full OpenML-style runs are expensive; keep a fixed large subsample for docs.
    rng = np.random.default_rng(42)
    n_sub = min(6_000, X_full.shape[0])
    idx = rng.choice(X_full.shape[0], size=n_sub, replace=False)
    X, y = X_full[idx], y_full[idx]
    model = SeparatedALSRegressor(
        rank=5,
        degree=4,
        basis="tent",
        ridge=0.007168023918105929,
        smoothness=7.218538035064866e-08,
        max_sweeps=100,
        tol=4.952524696642217e-06,
        n_init=3,
        refit_scales=True,
        fit_intercept=False,
        penalty_kind="tent_level",
        random_state=0,
        verbose=False,
    )
    model.fit(X, y)
    _save_separation(model, out_dir / "california_housing_separation_stages.png", n_grid=80)


def main() -> None:
    repo_root = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(description="Write README example PNGs under docs/images/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / DEFAULT_REL_IMAGES,
        help=f"Directory for PNG output (default: {DEFAULT_REL_IMAGES}).",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    example1_sklearn_friedman1(out_dir)
    example2_package_friedman1_p10(out_dir)
    example3_friedman2_tent(out_dir)
    example4_california_housing(out_dir)


if __name__ == "__main__":
    main()
