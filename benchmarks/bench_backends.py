from __future__ import annotations

import argparse
import statistics as stats
import time

import numpy as np

from sepals import SeparatedALSRegressor, friedman1


def _time_fit(X, y, *, backend: str, rank: int, degree: int, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        model = SeparatedALSRegressor(
            rank=rank,
            degree=degree,
            basis="legendre",
            max_sweeps=2,
            n_init=1,
            tol=0.0,
            random_state=123,
            kernel_backend=backend,
        )
        start = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - start)
    return stats.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ALS kernel backends.")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-mps", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(1234)
    X, y = friedman1(args.n, rng, p=args.p)

    backends = ["optimized"]
    if args.include_mps:
        # Warm the PyTorch import/device setup outside the measured region.
        SeparatedALSRegressor(
            rank=2,
            degree=2,
            basis="legendre",
            max_sweeps=1,
            n_init=1,
            kernel_backend="mps",
        ).fit(X[: min(args.n, 1000)], y[: min(args.n, 1000)])
        backends.append("mps")

    timings = {}
    for backend in backends:
        timings[backend] = _time_fit(
            X,
            y,
            backend=backend,
            rank=args.rank,
            degree=args.degree,
            repeats=args.repeats,
        )
        print(f"{backend:>9}: {timings[backend]:.4f}s")

    if "mps" in timings:
        print(f"speedup: {timings['optimized'] / timings['mps']:.3f}x")


if __name__ == "__main__":
    main()
