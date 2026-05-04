"""
Separated ALS regression following Beylkin, Garcke, and Mohlenkamp.

This module is an independent implementation of the separated-regression
algorithm described in "Multivariate Regression and Machine Learning with Sums
of Separable Functions" by Gregory Beylkin, Jochen Garcke, and Martin J.
Mohlenkamp.

Model:
    f(x) = intercept + sum_{l=1}^r s_l prod_{m=1}^d g_m^l(x_m)

Each one-dimensional factor g_m^l is expanded in a linear basis. Fitting uses
alternating least squares (ALS): fix all directions except m, collapse the
problem to a one-dimensional least-squares problem, solve for all rank terms in
that direction, normalize factors, and optionally refit the rank coefficients.

This variant keeps the same algorithm as `beylkin_als_replicate.py`, but reuses
large ALS work buffers inside each fit to reduce allocation and memory churn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _als_kernels as kernels
from ._mps_backend import fit_dense_mps

BasisKind = Literal["legendre", "monomial", "tent"]
KernelBackend = Literal["auto", "reference", "optimized", "mps"]


@dataclass
class _TentSparseInfo:
    x: np.ndarray
    x2: np.ndarray
    n_tents: int
    left_mask: np.ndarray
    right_mask: np.ndarray
    both_mask: np.ndarray
    left_idx: np.ndarray
    right_idx: np.ndarray
    both_left_idx: np.ndarray
    left_val: np.ndarray
    right_val: np.ndarray
    left_val2: np.ndarray
    right_val2: np.ndarray
    left_x: np.ndarray
    right_x: np.ndarray
    both_left_val: np.ndarray
    both_right_val: np.ndarray
    both_cross_val: np.ndarray
    tent_cols: np.ndarray
    offdiag_cols: np.ndarray


@dataclass(eq=False)
class SeparatedALSRegressor(RegressorMixin, BaseEstimator):
    rank: int = 4
    degree: int = 3
    basis: BasisKind = "legendre"
    ridge: float = 1e-8
    smoothness: float = 0.0
    penalty_kind: str = "degree2"  # degree, degree2, tent_level
    max_sweeps: int = 50
    tol: float = 1e-7
    n_init: int = 3
    random_state: Optional[int] = 0
    refit_scales: bool = True
    verbose: bool = False
    fit_intercept: bool = False
    kernel_backend: KernelBackend = "optimized"

    def _validate_hyperparameters(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if self.basis not in {"legendre", "monomial", "tent"}:
            raise ValueError("basis must be one of 'legendre', 'monomial', or 'tent'")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.smoothness < 0:
            raise ValueError("smoothness must be non-negative")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be at least 1")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")
        if self.kernel_backend not in {"auto", "reference", "optimized", "mps"}:
            raise ValueError("kernel_backend must be one of 'auto', 'reference', 'optimized', or 'mps'")

    def _scale_X_fit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.x_min_ = np.nanmin(X, axis=0)
        self.x_max_ = np.nanmax(X, axis=0)
        denom = self.x_max_ - self.x_min_
        denom[denom == 0] = 1.0
        self.x_range_ = denom
        Z = (X - self.x_min_) / self.x_range_
        return np.clip(Z, 0.0, 1.0)

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Z = (X - self.x_min_) / self.x_range_
        return np.clip(Z, 0.0, 1.0)

    def _basis_eval_1d(self, x: np.ndarray) -> np.ndarray:
        """Return basis matrix Phi with shape (n, M). x must be in [0, 1]."""
        x = np.asarray(x, dtype=float)
        if self.basis == "monomial":
            return np.vstack([x ** k for k in range(self.degree + 1)]).T
        if self.basis == "legendre":
            # Legendre basis on [-1, 1], more numerically stable than monomials.
            z = 2.0 * x - 1.0
            Phi = np.empty((x.shape[0], self.degree + 1), dtype=float)
            Phi[:, 0] = 1.0
            if self.degree >= 1:
                Phi[:, 1] = z
            for k in range(1, self.degree):
                Phi[:, k + 1] = ((2 * k + 1) * z * Phi[:, k] - k * Phi[:, k - 1]) / (k + 1)
            return Phi
        if self.basis == "tent":
            # Multilevel-like hat functions: constant, linear, then equally-spaced tents.
            # degree is interpreted as level. M = 2 + (2**level - 1).
            level = self.degree
            centers = np.linspace(0.0, 1.0, 2**level + 1)[1:-1] if level > 0 else np.array([])
            width = 1.0 / (2 ** max(level, 1))
            cols = [np.ones_like(x), x]
            for c in centers:
                cols.append(np.maximum(1.0 - np.abs(x - c) / width, 0.0))
            return np.vstack(cols).T
        raise ValueError(f"unknown basis {self.basis}")

    def _basis_matrices(self, Xs: np.ndarray):
        return [self._basis_eval_1d(Xs[:, j]) for j in range(Xs.shape[1])]

    def _penalty_diag(self, M: int) -> np.ndarray:
        # Regularization used in Beylkin et al. is basis-dependent.
        # For monomials on Friedman1 they say they penalize by degree.
        # For the multilevel tent basis on Friedman3 they use 0 for constant,
        # 1 for x, then doubling at each level. We expose both conventions.
        if self.smoothness <= 0:
            return np.full(M, self.ridge)
        if self.penalty_kind == "degree":
            w = np.arange(M, dtype=float)
        elif self.penalty_kind == "tent_level":
            # Assumes basis columns are [1, x, tents by increasing level].
            w = np.zeros(M, dtype=float)
            if M >= 2:
                w[1] = 1.0
            if M > 2:
                # columns 2: are tent functions; for a complete level L basis,
                # level q contributes 2^(q-1) tents and weight 2^q.
                idx = 2
                level = 1
                while idx < M:
                    count = 2 ** (level - 1)
                    w[idx:min(M, idx + count)] = 2.0 ** level
                    idx += count
                    level += 1
        else:
            w = np.arange(M, dtype=float) ** 2
        w[0] = 0.0
        return self.ridge + self.smoothness * w

    def _init_params(self, Phi_list, y, rng):
        p = len(Phi_list)
        M_list = [Phi.shape[1] for Phi in Phi_list]
        coeffs = []
        values = []
        for j, M in enumerate(M_list):
            C = rng.normal(0.0, 0.05, size=(self.rank, M))
            C[:, 0] += 1.0
            V = Phi_list[j] @ C.T
            # Normalize each factor empirically.
            norms = np.sqrt(np.mean(V**2, axis=0)) + 1e-12
            C = C / norms[:, None]
            V = V / norms[None, :]
            coeffs.append(C)
            values.append(V)
        # Least-squares initialization of scales for the random rank features.
        Q = np.ones((len(y), self.rank))
        for V in values:
            Q *= V
        s = np.linalg.lstsq(Q + 1e-12 * rng.normal(size=Q.shape), y, rcond=None)[0]
        return coeffs, values, s

    @staticmethod
    def _product_except(values, scales, skip: int) -> np.ndarray:
        return kernels.product_except_reference(values, scales, skip)

    @staticmethod
    def _product_except_into(values, scales, skip: int, out: np.ndarray) -> np.ndarray:
        return kernels.product_except_reference_into(values, scales, skip, out)

    @staticmethod
    def _rank_design(values) -> np.ndarray:
        return kernels.rank_design_reference(values)

    @staticmethod
    def _rank_design_into(values, out: np.ndarray) -> np.ndarray:
        return kernels.rank_design_reference_into(values, out)

    def _weighted_design_into(self, Phi: np.ndarray, P: np.ndarray, out: np.ndarray) -> np.ndarray:
        return kernels.weighted_design_reference_into(Phi, P, out)

    def _tent_sparse_info_1d(self, x: np.ndarray) -> _TentSparseInfo:
        n_tents = 2 ** self.degree - 1
        t = x * (2 ** self.degree)
        left_k = np.floor(t).astype(np.int64)
        frac = t - left_k
        right_k = left_k + 1
        left_mask = (left_k >= 1) & (left_k <= n_tents)
        right_mask = (right_k >= 1) & (right_k <= n_tents)
        both_mask = left_mask & right_mask
        return _TentSparseInfo(
            x=x,
            x2=x * x,
            n_tents=n_tents,
            left_mask=left_mask,
            right_mask=right_mask,
            both_mask=both_mask,
            left_idx=left_k[left_mask] - 1,
            right_idx=right_k[right_mask] - 1,
            both_left_idx=left_k[both_mask] - 1,
            left_val=1.0 - frac[left_mask],
            right_val=frac[right_mask],
            left_val2=(1.0 - frac[left_mask]) ** 2,
            right_val2=frac[right_mask] ** 2,
            left_x=x[left_mask],
            right_x=x[right_mask],
            both_left_val=1.0 - frac[both_mask],
            both_right_val=frac[both_mask],
            both_cross_val=(1.0 - frac[both_mask]) * frac[both_mask],
            tent_cols=np.arange(n_tents),
            offdiag_cols=np.arange(max(n_tents - 1, 0)),
        )

    @staticmethod
    def _tent_bincount(info: _TentSparseInfo, base: np.ndarray, power: int = 1) -> np.ndarray:
        return kernels.tent_bincount_reference(info, base, power)

    def _tent_normal_equations_into(
        self,
        info: _TentSparseInfo,
        P: np.ndarray,
        y: np.ndarray,
        lhs_out: np.ndarray,
        rhs_out: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return kernels.tent_normal_equations_reference_into(info, P, y, lhs_out, rhs_out)

    def _predict_from_params(self, Phi_list, coeffs, scales, intercept: float) -> np.ndarray:
        n = Phi_list[0].shape[0]
        Q = np.ones((n, self.rank))
        for j, Phi in enumerate(Phi_list):
            Q *= Phi @ coeffs[j].T
        return intercept + Q @ scales

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        self._validate_hyperparameters()
        X, y = validate_data(self, X, y, dtype=float, y_numeric=True)
        Xs = self._scale_X_fit(X)
        Phi_list = self._basis_matrices(Xs)
        max_M = max(Phi.shape[1] for Phi in Phi_list)
        use_sparse_tent = self.basis == "tent" and max_M >= 12
        tent_sparse_infos = (
            [self._tent_sparse_info_1d(Xs[:, j]) for j in range(Xs.shape[1])]
            if use_sparse_tent
            else None
        )
        self.intercept_ = float(np.mean(y)) if self.fit_intercept else 0.0
        yc = y - self.intercept_
        if X_val is not None:
            if y_val is None:
                raise ValueError("y_val must be provided when X_val is provided")
            X_val, y_val = validate_data(
                self,
                X_val,
                y_val,
                dtype=float,
                y_numeric=True,
                reset=False,
            )
            Xv = self._scale_X(X_val)
            Phi_val = self._basis_matrices(Xv)
        else:
            if y_val is not None:
                raise ValueError("X_val must be provided when y_val is provided")
            Phi_val = None

        rng_master = np.random.default_rng(self.random_state)
        if self.kernel_backend == "mps":
            if use_sparse_tent:
                raise ValueError(
                    "kernel_backend='mps' currently supports dense ALS paths only. "
                    "Use basis='legendre', basis='monomial', or a lower tent degree "
                    "that does not trigger the sparse tent path."
                )
            self.coeffs_, self.scales_, self.history_, self.all_histories_ = fit_dense_mps(
                self,
                Phi_list,
                yc,
                Phi_val,
                y_val,
                self.intercept_,
                rng_master,
            )
            return self

        best = None
        histories = []

        for init in range(self.n_init):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            coeffs, values, scales = self._init_params(Phi_list, yc, rng)
            prev_loss = np.inf
            history = []
            n = len(yc)
            use_optimized_products = self.kernel_backend in {"auto", "optimized"}
            use_large_basis_buffers = max_M >= 12
            use_rank_work = use_large_basis_buffers or use_optimized_products
            use_dense_work = not use_sparse_tent and use_rank_work
            P_work = np.empty((n, self.rank), dtype=float) if use_rank_work else None
            Q_work = np.empty((n, self.rank), dtype=float) if use_rank_work else None
            A_work = np.empty((n, self.rank * max_M), dtype=float) if use_dense_work else None
            lhs_work = (
                np.empty((self.rank * max_M, self.rank * max_M), dtype=float)
                if use_sparse_tent
                else None
            )
            rhs_work = np.empty(self.rank * max_M, dtype=float) if use_sparse_tent else None
            prefix_work = np.empty((n, self.rank), dtype=float) if use_optimized_products else None
            suffix_products = (
                [np.empty((n, self.rank), dtype=float) for _ in range(len(Phi_list) + 1)]
                if use_optimized_products
                else None
            )
            penalty_cache = {
                Phi.shape[1]: np.tile(self._penalty_diag(Phi.shape[1]), self.rank)
                for Phi in Phi_list
            }

            for sweep in range(self.max_sweeps):
                if use_optimized_products:
                    kernels.build_suffix_products(values, suffix_products)
                    prefix_work.fill(1.0)
                for m, Phi in enumerate(Phi_list):
                    n, M = Phi.shape
                    if use_optimized_products:
                        P = kernels.product_except_from_prefix_suffix(
                            scales,
                            prefix_work,
                            suffix_products[m + 1],
                            P_work,
                        )
                    elif use_large_basis_buffers:
                        P = self._product_except_into(values, scales, skip=m, out=P_work)
                    else:
                        P = self._product_except(values, scales, skip=m)
                    if use_sparse_tent:
                        if use_optimized_products:
                            lhs, rhs = kernels.tent_normal_equations_optimized_into(
                                tent_sparse_infos[m],
                                P,
                                yc,
                                lhs_work,
                                rhs_work,
                            )
                        else:
                            lhs, rhs = self._tent_normal_equations_into(
                                tent_sparse_infos[m],
                                P,
                                yc,
                                lhs_work,
                                rhs_work,
                            )
                    # A has blocks A_l = diag(P[:, l]) Phi.
                    elif use_dense_work:
                        lhs, rhs = kernels.dense_normal_equations_reference(
                            Phi,
                            P,
                            yc,
                            A_work[:, : self.rank * M],
                        )
                    else:
                        lhs, rhs = kernels.dense_normal_equations_reference(Phi, P, yc)
                    pen = penalty_cache[M]
                    lhs.flat[:: lhs.shape[0] + 1] += pen
                    try:
                        z = np.linalg.solve(lhs, rhs)
                    except np.linalg.LinAlgError:
                        z = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

                    C = z.reshape(self.rank, M)
                    V = Phi @ C.T
                    # Normalize factors and absorb scale into s_l.
                    norms = np.sqrt(np.mean(V**2, axis=0))
                    tiny = norms < 1e-12
                    if np.any(tiny):
                        # Reinitialize dead factors in this direction.
                        C[tiny, :] = rng.normal(0, 0.05, size=(tiny.sum(), M))
                        C[tiny, 0] += 1.0
                        V[:, tiny] = Phi @ C[tiny, :].T
                        norms[tiny] = np.sqrt(np.mean(V[:, tiny] ** 2, axis=0)) + 1e-12
                    C = C / norms[:, None]
                    V = V / norms[None, :]
                    scales = scales * norms
                    coeffs[m] = C
                    values[m] = V

                    if self.refit_scales:
                        if use_optimized_products:
                            Q = kernels.rank_design_from_prefix_suffix(
                                prefix_work,
                                V,
                                suffix_products[m + 1],
                                Q_work,
                            )
                        elif use_large_basis_buffers:
                            Q = self._rank_design_into(values, Q_work)
                        else:
                            Q = self._rank_design(values)
                        lhs_s = Q.T @ Q
                        lhs_s.flat[:: self.rank + 1] += self.ridge
                        rhs_s = Q.T @ yc
                        try:
                            scales = np.linalg.solve(lhs_s, rhs_s)
                        except np.linalg.LinAlgError:
                            scales = np.linalg.lstsq(lhs_s, rhs_s, rcond=None)[0]
                    if use_optimized_products:
                        prefix_work *= V

                if use_optimized_products:
                    pred_c = prefix_work @ scales
                elif use_large_basis_buffers:
                    pred_c = self._rank_design_into(values, Q_work) @ scales
                else:
                    pred_c = self._rank_design(values) @ scales
                train_mse = float(np.mean((yc - pred_c) ** 2))
                if Phi_val is not None:
                    pred_val = self._predict_from_params(Phi_val, coeffs, scales, self.intercept_)
                    val_mse = float(np.mean((y_val - pred_val) ** 2))
                else:
                    val_mse = np.nan
                history.append((train_mse, val_mse))
                if self.verbose:
                    print(f"init={init} sweep={sweep} train_mse={train_mse:.6g} val_mse={val_mse:.6g}")
                rel = (prev_loss - train_mse) / max(prev_loss, 1e-12)
                if rel >= 0 and rel < self.tol and sweep >= 3:
                    break
                prev_loss = train_mse

            histories.append(history)
            final_score = history[-1][1] if Phi_val is not None else history[-1][0]
            if best is None or final_score < best[0]:
                best = (final_score, coeffs, scales, history)

        self.coeffs_ = best[1]
        self.scales_ = best[2]
        self.history_ = best[3]
        self.all_histories_ = histories
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["coeffs_", "scales_", "x_min_", "x_range_"])
        X = validate_data(self, X, dtype=float, reset=False)
        Xs = self._scale_X(X)
        Phi_list = self._basis_matrices(Xs)
        return self._predict_from_params(Phi_list, self.coeffs_, self.scales_, self.intercept_)

    def factor_values(self, feature: int, grid: np.ndarray) -> np.ndarray:
        """Return g_feature^l(grid) for all l. grid is in original feature scale."""
        check_is_fitted(self, attributes=["coeffs_", "x_min_", "x_range_"])
        if feature < 0 or feature >= self.n_features_in_:
            raise ValueError(f"feature must be in [0, {self.n_features_in_})")
        z = (np.asarray(grid) - self.x_min_[feature]) / self.x_range_[feature]
        z = np.clip(z, 0, 1)
        Phi = self._basis_eval_1d(z)
        return Phi @ self.coeffs_[feature].T
