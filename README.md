# sepals

Separated ALS regression for scikit-learn, inspired by Beylkin, Garcke, and
Mohlenkamp.

The model has the separated form:

```text
f(x) = intercept + sum_l s_l prod_m g_m^l(x_m)
```

Each one-dimensional factor `g_m^l` is expanded in a basis and fitted with
alternating least squares. Supported bases are:

- `monomial`
- `legendre`
- `tent`

The `tent` basis uses a sparse normal-equation path for larger levels, avoiding
materializing the dense weighted design matrix.

## Attribution

This package implements a Python/scikit-learn style version of the separated
regression algorithm described in:

> Gregory Beylkin, Jochen Garcke, and Martin J. Mohlenkamp.
> "Multivariate Regression and Machine Learning with Sums of Separable
> Functions." Submitted December 2007; revised September 2008.
> PDF: <https://ins.uni-bonn.de/media/public/publication-media/BE-GA-MO2007P.pdf?pk=625>

The separated representation, alternating least-squares fitting strategy,
one-dimensional normal equations, smoothness regularization idea, and multilevel
tent basis are due to Beylkin, Garcke, and Mohlenkamp. This package is an
independent implementation for experimentation and reproduction; it is not
affiliated with or endorsed by the paper authors.

The package license covers only this implementation code and documentation. It
does not relicense the original paper or any third-party datasets, benchmarks,
or text.

Implementation note: this entire codebase was written by GPT-5.5.

## Install

From this directory:

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[test]"
pytest
```

You can also run tests without installing permanently:

```bash
uv run --with pytest pytest
```

## Quick Start

```python
import numpy as np
from sepals import SeparatedALSRegressor, friedman1, rmse

rng = np.random.default_rng(123)
X_train, y_train = friedman1(2_000, rng, p=10)
X_test, y_test = friedman1(500, rng, p=10)

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

pred = model.predict(X_test)
print("RMSE:", rmse(y_test, pred))
```

## Scikit-Learn Usage

`SeparatedALSRegressor` follows the scikit-learn estimator API. It supports
`get_params`, `set_params`, `score`, `Pipeline`, `GridSearchCV`, and
`sklearn.base.clone`.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sepals import SeparatedALSRegressor, friedman1

rng = np.random.default_rng(123)
X, y = friedman1(300, rng, p=6)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("als", SeparatedALSRegressor(
        basis="legendre",
        max_sweeps=8,
        n_init=1,
        random_state=123,
        fit_intercept=True,
    )),
])

search = GridSearchCV(
    pipe,
    {
        "als__rank": [1, 2],
        "als__degree": [2, 3],
    },
    cv=3,
)
search.fit(X, y)
print(search.best_params_)
print(search.score(X, y))
```

## Tent Basis Example

```python
import numpy as np
from sepals import SeparatedALSRegressor, friedman2

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
```

## API

### `SeparatedALSRegressor`

Main estimator.

Important parameters:

- `rank`: number of separated rank terms.
- `degree`: polynomial degree, or tent level for `basis="tent"`.
- `basis`: one of `"monomial"`, `"legendre"`, or `"tent"`.
- `ridge`: small diagonal regularization.
- `smoothness`: basis coefficient smoothness penalty.
- `penalty_kind`: `"degree"`, `"degree2"`, or `"tent_level"`.
- `max_sweeps`: maximum ALS sweeps per random initialization.
- `tol`: relative training-loss stopping tolerance.
- `n_init`: number of random initializations.
- `fit_intercept`: whether to subtract/add a mean intercept.

Methods:

- `fit(X, y, X_val=None, y_val=None)`
- `predict(X)`
- `factor_values(feature, grid)`

### Dataset Helpers

- `friedman1(n, rng, noise_std=0.0, p=10)`
- `friedman2(n, rng, noise_std=0.0)`
- `rmse(y, yhat)`

## Notes

This package keeps the implementation NumPy-only. The biggest remaining cost is
the repeated ALS fitting across hyperparameter grids. For large grid searches,
parallelize at the experiment level.
