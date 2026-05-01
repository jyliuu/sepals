import numpy as np

from sepals.als import SeparatedALSRegressor


def _dense_normal_equations(model, phi, p, y):
    n, m = phi.shape
    a = np.empty((n, model.rank * m), dtype=float)
    for rank_idx in range(model.rank):
        a[:, rank_idx * m:(rank_idx + 1) * m] = p[:, [rank_idx]] * phi
    return a.T @ a, a.T @ y


def test_sparse_tent_normal_equations_match_dense_reference():
    rng = np.random.default_rng(123)
    n = 300
    model = SeparatedALSRegressor(rank=3, degree=5, basis="tent")
    x = rng.uniform(0.0, 1.0, size=n)
    phi = model._basis_eval_1d(x)
    p = rng.normal(size=(n, model.rank))
    y = rng.normal(size=n)
    info = model._tent_sparse_info_1d(x)
    lhs_work = np.empty((model.rank * phi.shape[1], model.rank * phi.shape[1]), dtype=float)
    rhs_work = np.empty(model.rank * phi.shape[1], dtype=float)

    dense_lhs, dense_rhs = _dense_normal_equations(model, phi, p, y)
    sparse_lhs, sparse_rhs = model._tent_normal_equations_into(
        info,
        p,
        y,
        lhs_work,
        rhs_work,
    )

    np.testing.assert_allclose(sparse_lhs, dense_lhs, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(sparse_rhs, dense_rhs, rtol=1e-11, atol=1e-10)
