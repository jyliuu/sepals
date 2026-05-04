import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sepals import SeparatedALSRegressor, friedman1
from sepals.plotting import plot_separation_stages, plot_single_stage


def test_plot_separation_stages_saves_nonempty():
    rng = np.random.default_rng(0)
    X, y = friedman1(80, rng, p=5)
    model = SeparatedALSRegressor(
        rank=2,
        degree=2,
        basis="legendre",
        max_sweeps=3,
        tol=0.0,
        n_init=1,
        random_state=0,
    ).fit(X, y)

    fig = plot_separation_stages(model, n_grid=50)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    assert len(buf.getvalue()) > 500


def test_plot_single_stage_saves_nonempty():
    rng = np.random.default_rng(1)
    X, y = friedman1(70, rng, p=5)
    model = SeparatedALSRegressor(
        rank=3,
        degree=2,
        basis="monomial",
        max_sweeps=2,
        tol=0.0,
        n_init=1,
        random_state=1,
    ).fit(X, y)

    fig = plot_single_stage(model, 1, n_grid=40)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    assert len(buf.getvalue()) > 500
