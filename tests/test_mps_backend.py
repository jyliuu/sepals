import numpy as np
import pytest

from sepals import SeparatedALSRegressor, friedman1


def _mps_available():
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def test_mps_backend_reports_missing_optional_runtime():
    if _mps_available():
        pytest.skip("MPS is available; parity test covers this environment")

    rng = np.random.default_rng(123)
    X, y = friedman1(40, rng, p=5)
    model = SeparatedALSRegressor(
        rank=2,
        degree=2,
        basis="legendre",
        max_sweeps=1,
        n_init=1,
        kernel_backend="mps",
    )

    with pytest.raises((ImportError, RuntimeError), match="MPS|PyTorch|mps"):
        model.fit(X, y)


def test_mps_backend_rejects_sparse_tent_path_before_runtime_check():
    rng = np.random.default_rng(456)
    X, y = friedman1(40, rng, p=5)
    model = SeparatedALSRegressor(
        rank=2,
        degree=4,
        basis="tent",
        max_sweeps=1,
        n_init=1,
        kernel_backend="mps",
    )

    with pytest.raises(ValueError, match="dense ALS paths only"):
        model.fit(X, y)


@pytest.mark.skipif(not _mps_available(), reason="PyTorch MPS is not available")
def test_mps_backend_predictions_match_optimized_cpu_reasonably():
    rng = np.random.default_rng(789)
    X, y = friedman1(120, rng, p=5)
    kwargs = dict(
        rank=2,
        degree=3,
        basis="legendre",
        max_sweeps=3,
        tol=0.0,
        n_init=1,
        random_state=123,
        fit_intercept=True,
    )

    cpu = SeparatedALSRegressor(**kwargs, kernel_backend="optimized").fit(X[:90], y[:90])
    mps = SeparatedALSRegressor(**kwargs, kernel_backend="mps").fit(X[:90], y[:90])

    np.testing.assert_allclose(
        mps.predict(X[90:]),
        cpu.predict(X[90:]),
        rtol=5e-2,
        atol=5e-2,
    )
