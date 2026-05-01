import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from sepals import SeparatedALSRegressor, friedman1


def test_get_set_params_and_clone():
    model = SeparatedALSRegressor(rank=2, degree=3, basis="legendre")

    params = model.get_params()
    assert params["rank"] == 2
    assert params["degree"] == 3
    assert params["basis"] == "legendre"

    model.set_params(rank=4)
    assert model.rank == 4

    cloned = clone(model)
    assert isinstance(cloned, SeparatedALSRegressor)
    assert cloned.rank == 4
    assert cloned.degree == 3


def test_score_and_fit_state_follow_sklearn_conventions():
    rng = np.random.default_rng(20)
    X, y = friedman1(140, rng, p=5)
    model = SeparatedALSRegressor(
        rank=2,
        degree=2,
        basis="legendre",
        max_sweeps=4,
        tol=0.0,
        n_init=1,
        random_state=123,
        fit_intercept=True,
    )

    model.fit(X[:100], y[:100])

    check_is_fitted(model)
    assert model.n_features_in_ == X.shape[1]
    assert np.isfinite(model.score(X[100:], y[100:]))


def test_pipeline_and_grid_search_work():
    rng = np.random.default_rng(21)
    X, y = friedman1(120, rng, p=5)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "als",
                SeparatedALSRegressor(
                    basis="legendre",
                    max_sweeps=3,
                    tol=0.0,
                    n_init=1,
                    random_state=123,
                    fit_intercept=True,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipe,
        {
            "als__rank": [1, 2],
            "als__degree": [1, 2],
        },
        cv=3,
        error_score="raise",
    )

    search.fit(X, y)

    assert "als__rank" in search.best_params_
    assert np.isfinite(search.score(X, y))


def test_sklearn_estimator_checks_pass_for_representative_config():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping check check_regressor_data_not_an_array.*",
        )
        check_estimator(
            SeparatedALSRegressor(
                rank=4,
                degree=3,
                basis="legendre",
                max_sweeps=20,
                n_init=2,
                random_state=0,
                fit_intercept=True,
            )
        )
