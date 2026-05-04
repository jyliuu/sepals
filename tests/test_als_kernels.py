import numpy as np

from sepals import SeparatedALSRegressor, friedman1
from sepals import _als_kernels as kernels


def test_prefix_suffix_product_kernels_match_reference():
    rng = np.random.default_rng(123)
    values = [rng.normal(size=(50, 4)) for _ in range(5)]
    scales = rng.normal(size=4)
    suffix_products = [np.empty_like(values[0]) for _ in range(len(values) + 1)]
    prefix = np.empty_like(values[0])
    out = np.empty_like(values[0])

    kernels.build_suffix_products(values, suffix_products)
    prefix.fill(1.0)
    for skip, value in enumerate(values):
        expected = kernels.product_except_reference(values, scales, skip)
        actual = kernels.product_except_from_prefix_suffix(
            scales,
            prefix,
            suffix_products[skip + 1],
            out,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

        rank_expected = kernels.rank_design_reference(values)
        rank_actual = kernels.rank_design_from_prefix_suffix(
            prefix,
            value,
            suffix_products[skip + 1],
            out,
        )
        np.testing.assert_allclose(rank_actual, rank_expected, rtol=1e-14, atol=1e-14)
        prefix *= value


def test_dense_normal_equations_kernel_matches_manual_reference():
    rng = np.random.default_rng(456)
    phi = rng.normal(size=(80, 6))
    p = rng.normal(size=(80, 3))
    y = rng.normal(size=80)
    design = np.empty((80, 18), dtype=float)
    for rank_idx in range(p.shape[1]):
        start = rank_idx * phi.shape[1]
        design[:, start : start + phi.shape[1]] = p[:, [rank_idx]] * phi

    lhs, rhs = kernels.dense_normal_equations_reference(phi, p, y)

    np.testing.assert_allclose(lhs, design.T @ design)
    np.testing.assert_allclose(rhs, design.T @ y)


def test_optimized_tent_kernel_matches_reference_kernel():
    rng = np.random.default_rng(654)
    model = SeparatedALSRegressor(rank=4, degree=5, basis="tent")
    x = rng.uniform(size=400)
    p = rng.normal(size=(400, model.rank))
    y = rng.normal(size=400)
    info = model._tent_sparse_info_1d(x)
    m = info.n_tents + 2
    lhs_ref = np.empty((model.rank * m, model.rank * m), dtype=float)
    rhs_ref = np.empty(model.rank * m, dtype=float)
    lhs_opt = np.empty_like(lhs_ref)
    rhs_opt = np.empty_like(rhs_ref)

    expected = kernels.tent_normal_equations_reference_into(info, p, y, lhs_ref, rhs_ref)
    actual = kernels.tent_normal_equations_optimized_into(info, p, y, lhs_opt, rhs_opt)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-10)


def test_optimized_backend_matches_reference_backend_predictions():
    rng = np.random.default_rng(789)
    X, y = friedman1(180, rng, p=5)
    kwargs = dict(
        rank=2,
        degree=3,
        basis="legendre",
        max_sweeps=5,
        tol=0.0,
        n_init=1,
        random_state=123,
        fit_intercept=True,
    )

    reference = SeparatedALSRegressor(**kwargs, kernel_backend="reference").fit(X[:140], y[:140])
    optimized = SeparatedALSRegressor(**kwargs, kernel_backend="optimized").fit(X[:140], y[:140])

    np.testing.assert_allclose(
        optimized.predict(X[140:]),
        reference.predict(X[140:]),
        rtol=1e-10,
        atol=1e-10,
    )
