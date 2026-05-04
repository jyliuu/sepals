"""Internal kernels for separated ALS fitting.

The functions in this module keep hot, swappable numerical kernels out of the
estimator class. Reference kernels preserve the original NumPy implementation;
optimized kernels can use different product or normal-equation strategies while
being tested against the reference path.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def product_except_reference(
    values: Sequence[np.ndarray],
    scales: np.ndarray,
    skip: int,
) -> np.ndarray:
    n = values[0].shape[0]
    out = np.tile(scales, (n, 1)).astype(float)
    return product_except_reference_into(values, scales, skip, out)


def product_except_reference_into(
    values: Sequence[np.ndarray],
    scales: np.ndarray,
    skip: int,
    out: np.ndarray,
) -> np.ndarray:
    out[...] = scales
    for j, value in enumerate(values):
        if j != skip:
            out *= value
    return out


def rank_design_reference(values: Sequence[np.ndarray]) -> np.ndarray:
    out = np.ones_like(values[0])
    return rank_design_reference_into(values, out)


def rank_design_reference_into(
    values: Sequence[np.ndarray],
    out: np.ndarray,
) -> np.ndarray:
    out.fill(1.0)
    for value in values:
        out *= value
    return out


def build_suffix_products(
    values: Sequence[np.ndarray],
    suffix_products: Sequence[np.ndarray],
) -> Sequence[np.ndarray]:
    """Fill suffix_products[j] with prod(values[j:])."""
    suffix_products[-1].fill(1.0)
    for j in range(len(values) - 1, -1, -1):
        np.multiply(suffix_products[j + 1], values[j], out=suffix_products[j])
    return suffix_products


def product_except_from_prefix_suffix(
    scales: np.ndarray,
    prefix_product: np.ndarray,
    suffix_product: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    np.multiply(prefix_product, suffix_product, out=out)
    out *= scales
    return out


def rank_design_from_prefix_suffix(
    prefix_product: np.ndarray,
    current_value: np.ndarray,
    suffix_product: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    np.multiply(prefix_product, current_value, out=out)
    out *= suffix_product
    return out


def weighted_design_reference_into(
    phi: np.ndarray,
    p: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    n, m = phi.shape
    rank = p.shape[1]
    design = out[:, : rank * m]
    if m >= 12:
        design.reshape(n, rank, m)[:] = p[:, :, None] * phi[:, None, :]
    else:
        for rank_idx in range(rank):
            start = rank_idx * m
            design[:, start : start + m] = p[:, [rank_idx]] * phi
    return design


def dense_normal_equations_reference(
    phi: np.ndarray,
    p: np.ndarray,
    y: np.ndarray,
    design_work: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n, m = phi.shape
    rank = p.shape[1]
    if design_work is None:
        design_work = np.empty((n, rank * m), dtype=float)
    design = weighted_design_reference_into(phi, p, design_work)
    return design.T @ design, design.T @ y


def tent_bincount_reference(info, base: np.ndarray, power: int = 1) -> np.ndarray:
    if power == 1:
        left_weights = base[info.left_mask] * info.left_val
        right_weights = base[info.right_mask] * info.right_val
    else:
        left_weights = base[info.left_mask] * info.left_val * info.left_val
        right_weights = base[info.right_mask] * info.right_val * info.right_val
    return (
        np.bincount(info.left_idx, weights=left_weights, minlength=info.n_tents)
        + np.bincount(info.right_idx, weights=right_weights, minlength=info.n_tents)
    )


def tent_bincount_optimized(info, base: np.ndarray, power: int = 1) -> np.ndarray:
    if power == 1:
        left_values = info.left_val
        right_values = info.right_val
    else:
        left_values = info.left_val2
        right_values = info.right_val2
    return (
        np.bincount(
            info.left_idx,
            weights=base[info.left_mask] * left_values,
            minlength=info.n_tents,
        )
        + np.bincount(
            info.right_idx,
            weights=base[info.right_mask] * right_values,
            minlength=info.n_tents,
        )
    )


def tent_bincount_x_optimized(info, base: np.ndarray) -> np.ndarray:
    return (
        np.bincount(
            info.left_idx,
            weights=base[info.left_mask] * info.left_x * info.left_val,
            minlength=info.n_tents,
        )
        + np.bincount(
            info.right_idx,
            weights=base[info.right_mask] * info.right_x * info.right_val,
            minlength=info.n_tents,
        )
    )


def tent_normal_equations_reference_into(
    info,
    p: np.ndarray,
    y: np.ndarray,
    lhs_out: np.ndarray,
    rhs_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rank = p.shape[1]
    m = info.n_tents + 2
    width = rank * m
    lhs = lhs_out[:width, :width]
    rhs = rhs_out[:width]
    lhs.fill(0.0)

    for left_rank in range(rank):
        row = left_rank * m
        wy = p[:, left_rank] * y
        rhs[row] = np.sum(wy)
        rhs[row + 1] = wy @ info.x
        rhs[row + 2 : row + m] = tent_bincount_reference(info, wy)

        for right_rank in range(left_rank, rank):
            col = right_rank * m
            block = lhs[row : row + m, col : col + m]
            w = p[:, left_rank] * p[:, right_rank]
            sum_w = np.sum(w)
            sum_wx = w @ info.x
            block[0, 0] = sum_w
            block[0, 1] = sum_wx
            block[1, 0] = sum_wx
            block[1, 1] = w @ info.x2

            const_tent = tent_bincount_reference(info, w)
            x_tent = tent_bincount_reference(info, w * info.x)
            tent_diag = tent_bincount_reference(info, w, power=2)
            block[0, 2:m] = const_tent
            block[2:m, 0] = const_tent
            block[1, 2:m] = x_tent
            block[2:m, 1] = x_tent

            tent_cols = np.arange(info.n_tents)
            block[2 + tent_cols, 2 + tent_cols] = tent_diag
            if info.n_tents > 1:
                offdiag = np.bincount(
                    info.both_left_idx,
                    weights=(
                        w[info.both_mask]
                        * info.both_left_val
                        * info.both_right_val
                    ),
                    minlength=info.n_tents,
                )[: info.n_tents - 1]
                left_cols = np.arange(info.n_tents - 1)
                block[2 + left_cols, 3 + left_cols] = offdiag
                block[3 + left_cols, 2 + left_cols] = offdiag

            if right_rank != left_rank:
                lhs[col : col + m, row : row + m] = block.T

    return lhs, rhs


def tent_normal_equations_optimized_into(
    info,
    p: np.ndarray,
    y: np.ndarray,
    lhs_out: np.ndarray,
    rhs_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rank = p.shape[1]
    n = p.shape[0]
    m = info.n_tents + 2
    width = rank * m
    lhs = lhs_out[:width, :width]
    rhs = rhs_out[:width]
    lhs.fill(0.0)

    wy_work = np.empty(n, dtype=float)
    w_work = np.empty(n, dtype=float)

    for left_rank in range(rank):
        row = left_rank * m
        np.multiply(p[:, left_rank], y, out=wy_work)
        rhs[row] = np.sum(wy_work)
        rhs[row + 1] = wy_work @ info.x
        rhs[row + 2 : row + m] = tent_bincount_optimized(info, wy_work)

        for right_rank in range(left_rank, rank):
            col = right_rank * m
            block = lhs[row : row + m, col : col + m]
            np.multiply(p[:, left_rank], p[:, right_rank], out=w_work)
            sum_w = np.sum(w_work)
            sum_wx = w_work @ info.x
            block[0, 0] = sum_w
            block[0, 1] = sum_wx
            block[1, 0] = sum_wx
            block[1, 1] = w_work @ info.x2

            const_tent = tent_bincount_optimized(info, w_work)
            x_tent = tent_bincount_x_optimized(info, w_work)
            tent_diag = tent_bincount_optimized(info, w_work, power=2)
            block[0, 2:m] = const_tent
            block[2:m, 0] = const_tent
            block[1, 2:m] = x_tent
            block[2:m, 1] = x_tent

            block[2 + info.tent_cols, 2 + info.tent_cols] = tent_diag
            if info.n_tents > 1:
                offdiag = np.bincount(
                    info.both_left_idx,
                    weights=w_work[info.both_mask] * info.both_cross_val,
                    minlength=info.n_tents,
                )[: info.n_tents - 1]
                block[2 + info.offdiag_cols, 3 + info.offdiag_cols] = offdiag
                block[3 + info.offdiag_cols, 2 + info.offdiag_cols] = offdiag

            if right_rank != left_rank:
                lhs[col : col + m, row : row + m] = block.T

    return lhs, rhs
