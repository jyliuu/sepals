import numpy as np

from sepals import SeparatedALSRegressor, friedman1, friedman2, rmse


def test_monomial_fit_predict_is_deterministic():
    rng = np.random.default_rng(10)
    X, y = friedman1(180, rng, p=5)
    kwargs = dict(
        rank=2,
        degree=3,
        basis="monomial",
        max_sweeps=6,
        tol=0.0,
        n_init=2,
        random_state=123,
        fit_intercept=True,
    )

    first = SeparatedALSRegressor(**kwargs).fit(X[:130], y[:130])
    second = SeparatedALSRegressor(**kwargs).fit(X[:130], y[:130])

    np.testing.assert_allclose(first.predict(X[130:]), second.predict(X[130:]))
    assert rmse(y[130:], first.predict(X[130:])) >= 0.0


def test_legendre_fit_with_validation_records_history():
    rng = np.random.default_rng(11)
    X, y = friedman1(160, rng, p=5)
    model = SeparatedALSRegressor(
        rank=2,
        degree=3,
        basis="legendre",
        max_sweeps=5,
        tol=0.0,
        n_init=1,
        random_state=123,
        fit_intercept=True,
    )

    model.fit(X[:100], y[:100], X[100:130], y[100:130])

    assert len(model.history_) == 5
    assert np.asarray(model.history_).shape == (5, 2)
    assert model.predict(X[130:]).shape == (30,)


def test_tent_sparse_path_fits_and_predicts():
    rng = np.random.default_rng(12)
    X, y = friedman2(260, rng)
    model = SeparatedALSRegressor(
        rank=3,
        degree=4,
        basis="tent",
        ridge=1e-8,
        smoothness=1e-6,
        penalty_kind="tent_level",
        max_sweeps=5,
        tol=0.0,
        n_init=1,
        random_state=123,
        fit_intercept=True,
    )

    model.fit(X[:180], y[:180], X[180:220], y[180:220])
    pred = model.predict(X[220:])
    factors = model.factor_values(0, np.linspace(X[:, 0].min(), X[:, 0].max(), 25))

    assert np.all(np.isfinite(pred))
    assert factors.shape == (25, 3)
