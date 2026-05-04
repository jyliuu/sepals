"""PyTorch MPS backend for dense separated ALS fits.

This backend targets Apple Silicon through PyTorch's ``mps`` device. It is kept
optional so importing sepals does not require PyTorch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _import_torch_mps():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "kernel_backend='mps' requires PyTorch. Install the optional MPS "
            "dependency with `pip install sepals[mps]` or install torch manually."
        ) from exc

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            reason = "this PyTorch install was not built with MPS enabled"
        else:
            reason = "this machine does not expose an available MPS device"
        raise RuntimeError(f"kernel_backend='mps' requested, but {reason}.")
    return torch


def _torch_predict_from_params(torch, phi_list, coeffs, scales, intercept: float):
    q = torch.ones(
        (phi_list[0].shape[0], scales.shape[0]),
        dtype=phi_list[0].dtype,
        device=phi_list[0].device,
    )
    for phi, coeff in zip(phi_list, coeffs):
        q *= phi @ coeff.T
    return intercept + q @ scales


def _solve_system(torch, lhs, rhs, device, dtype):
    """Solve a small normal-equation system, with CPU fallback for MPS gaps."""
    try:
        return torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        lhs_cpu = lhs.detach().cpu()
        rhs_cpu = rhs.detach().cpu()
        try:
            solution = torch.linalg.solve(lhs_cpu, rhs_cpu)
        except RuntimeError:
            solution = torch.linalg.lstsq(lhs_cpu, rhs_cpu).solution
        return solution.to(device=device, dtype=dtype)


def fit_dense_mps(
    model,
    phi_list_np,
    yc_np: np.ndarray,
    phi_val_np: Optional[list[np.ndarray]],
    y_val_np: Optional[np.ndarray],
    intercept: float,
    rng_master: np.random.Generator,
):
    """Run dense ALS on Apple Silicon MPS and return fit state.

    MPS does not support NumPy-style float64 workflows broadly, so this path uses
    float32 on device and copies final coefficients back to NumPy.
    """
    torch = _import_torch_mps()
    device = torch.device("mps")
    dtype = torch.float32

    phi_list = [
        torch.as_tensor(np.asarray(phi, dtype=np.float32), dtype=dtype, device=device)
        for phi in phi_list_np
    ]
    yc = torch.as_tensor(np.asarray(yc_np, dtype=np.float32), dtype=dtype, device=device)
    if phi_val_np is not None:
        phi_val = [
            torch.as_tensor(np.asarray(phi, dtype=np.float32), dtype=dtype, device=device)
            for phi in phi_val_np
        ]
        y_val = torch.as_tensor(np.asarray(y_val_np, dtype=np.float32), dtype=dtype, device=device)
    else:
        phi_val = None
        y_val = None

    penalty_cache = {
        phi.shape[1]: torch.as_tensor(
            np.tile(model._penalty_diag(phi.shape[1]), model.rank).astype(np.float32),
            dtype=dtype,
            device=device,
        )
        for phi in phi_list_np
    }
    histories = []
    best = None

    with torch.inference_mode():
        for init in range(model.n_init):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            coeffs_np, values_np, scales_np = model._init_params(phi_list_np, yc_np, rng)
            coeffs = [
                torch.as_tensor(np.asarray(coeff, dtype=np.float32), dtype=dtype, device=device)
                for coeff in coeffs_np
            ]
            values = [
                torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=dtype, device=device)
                for value in values_np
            ]
            scales = torch.as_tensor(np.asarray(scales_np, dtype=np.float32), dtype=dtype, device=device)
            suffix_products = [torch.empty_like(values[0]) for _ in range(len(values) + 1)]
            prefix = torch.empty_like(values[0])
            p_work = torch.empty_like(values[0])
            q_work = torch.empty_like(values[0])
            max_m = max(phi.shape[1] for phi in phi_list)
            max_width = model.rank * max_m
            weighted_work = torch.empty(
                (yc.shape[0], model.rank, max_m),
                dtype=dtype,
                device=device,
            )
            lhs_work = torch.empty((max_width, max_width), dtype=dtype, device=device)
            rhs_work = torch.empty(max_width, dtype=dtype, device=device)
            lhs_scale_work = torch.empty((model.rank, model.rank), dtype=dtype, device=device)
            rhs_scale_work = torch.empty(model.rank, dtype=dtype, device=device)
            pred_work = torch.empty(yc.shape[0], dtype=dtype, device=device)
            prev_loss = np.inf
            history = []

            for sweep in range(model.max_sweeps):
                suffix_products[-1].fill_(1.0)
                for j in range(len(values) - 1, -1, -1):
                    torch.mul(suffix_products[j + 1], values[j], out=suffix_products[j])
                prefix.fill_(1.0)

                for feature_idx, phi in enumerate(phi_list):
                    m = phi.shape[1]
                    torch.mul(prefix, suffix_products[feature_idx + 1], out=p_work)
                    p_work *= scales
                    weighted = weighted_work[:, :, :m]
                    torch.mul(p_work[:, :, None], phi[:, None, :], out=weighted)
                    design = weighted.reshape(yc.shape[0], model.rank * m)
                    width = model.rank * m
                    lhs = lhs_work[:width, :width]
                    rhs = rhs_work[:width]
                    torch.mm(design.T, design, out=lhs)
                    torch.mv(design.T, yc, out=rhs)
                    lhs.diagonal().add_(penalty_cache[m])
                    z = _solve_system(torch, lhs, rhs, device, dtype)

                    c = z.reshape(model.rank, m)
                    v = values[feature_idx]
                    torch.mm(phi, c.T, out=v)
                    norms = torch.sqrt(torch.mean(v * v, dim=0))
                    tiny = norms < 1e-12
                    if bool(torch.any(tiny).item()):
                        tiny_idx = torch.nonzero(tiny).flatten().cpu().numpy()
                        c_reinit = rng.normal(0, 0.05, size=(len(tiny_idx), m)).astype(np.float32)
                        c_reinit[:, 0] += 1.0
                        c[tiny_idx, :] = torch.as_tensor(c_reinit, dtype=dtype, device=device)
                        torch.mm(phi, c.T, out=v)
                        norms[tiny_idx] = torch.sqrt(torch.mean(v[:, tiny_idx] ** 2, dim=0)) + 1e-12
                    c /= norms[:, None]
                    v /= norms[None, :]
                    scales *= norms
                    coeffs[feature_idx] = c

                    if model.refit_scales:
                        torch.mul(prefix, v, out=q_work)
                        q_work *= suffix_products[feature_idx + 1]
                        lhs_s = lhs_scale_work
                        rhs_s = rhs_scale_work
                        torch.mm(q_work.T, q_work, out=lhs_s)
                        lhs_s.diagonal().add_(model.ridge)
                        torch.mv(q_work.T, yc, out=rhs_s)
                        scales = _solve_system(torch, lhs_s, rhs_s, device, dtype)
                    prefix *= v

                torch.mv(prefix, scales, out=pred_work)
                pred_c = pred_work
                train_mse = float(torch.mean((yc - pred_c) ** 2).cpu().item())
                if phi_val is not None:
                    pred_val = _torch_predict_from_params(torch, phi_val, coeffs, scales, intercept)
                    val_mse = float(torch.mean((y_val - pred_val) ** 2).cpu().item())
                else:
                    val_mse = np.nan
                history.append((train_mse, val_mse))
                if model.verbose:
                    print(f"init={init} sweep={sweep} train_mse={train_mse:.6g} val_mse={val_mse:.6g}")
                rel = (prev_loss - train_mse) / max(prev_loss, 1e-12)
                if rel >= 0 and rel < model.tol and sweep >= 3:
                    break
                prev_loss = train_mse

            coeffs_out = [coeff.detach().cpu().numpy().astype(float, copy=False) for coeff in coeffs]
            scales_out = scales.detach().cpu().numpy().astype(float, copy=False)
            histories.append(history)
            final_score = history[-1][1] if phi_val is not None else history[-1][0]
            if best is None or final_score < best[0]:
                best = (final_score, coeffs_out, scales_out, history)

    return best[1], best[2], best[3], histories
