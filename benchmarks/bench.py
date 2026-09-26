"""Timing and peak memory of the estimators, against a direct NumPy implementation.

Run with e.g.

    python benchmarks/bench.py --nmesh 256 512 --nthreads 8
"""

import argparse
import resource
import time

import numpy as np

import field2pt as f2


def numpy_power(field, lbox, kedges, method):
    n = field.shape
    fh = np.fft.rfftn(field.astype(np.float64))
    i = [np.arange(m) for m in n]
    mx, my = (
        np.where(a <= m // 2, a, a - m).astype(np.float64) for a, m in zip(i[:2], n[:2])
    )
    mz = np.arange(n[2] // 2 + 1).astype(np.float64)
    MX, MY, MZ = np.meshgrid(mx, my, mz, indexing="ij")
    K = np.sqrt(
        (MX * (2 * np.pi / lbox[0])) ** 2
        + (MY * (2 * np.pi / lbox[1])) ** 2
        + (MZ * (2 * np.pi / lbox[2])) ** 2
    )
    p = fh.real**2 + fh.imag**2
    if method > 0:

        def sinc(m, nn):
            x = np.pi * m / nn
            return np.where(
                m == 0, 1.0, np.sin(np.where(m == 0, 1.0, x)) / np.where(m == 0, 1.0, x)
            )

        w = sinc(MX, n[0]) * sinc(MY, n[1]) * sinc(MZ, n[2])
        p = p / w ** (2 * method)
    keep = np.ones(MZ.shape, bool)
    plane = np.zeros(n[2] // 2 + 1, bool)
    plane[0] = True
    if n[2] % 2 == 0:
        plane[-1] = True
    red = (MX < 0) | (((MX == 0) | ((n[0] % 2 == 0) & (MX == n[0] // 2))) & (MY < 0))
    keep[red & plane[None, None, :]] = False
    nbin = len(kedges) - 1
    idx = np.digitize(K.ravel(), kedges) - 1
    m = (idx >= 0) & (idx < nbin) & keep.ravel()
    nm = np.bincount(idx[m], None, nbin)
    norm = np.prod(lbox) / np.prod(n) ** 2
    return norm * np.bincount(idx[m], p.ravel()[m], nbin) / nm


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def timed(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmesh", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--nthreads", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument(
        "--numpy", action="store_true", help="also time the NumPy reference"
    )
    args = ap.parse_args()

    dt = np.float32 if args.dtype == "float32" else np.float64
    rng = np.random.default_rng(0)

    print(f"dtype={args.dtype}  nthreads={args.nthreads}  reps={args.reps}")
    print(
        f"{'nmesh':>7s} {'field':>9s} {'P(k)':>9s} {'xi(s)':>9s} {'inter':>9s} "
        f"{'smooth':>9s} {'numpy P(k)':>11s} {'peak RSS':>10s}"
    )

    for n in args.nmesh:
        f = rng.standard_normal((n, n, n), dtype=np.float64).astype(dt)
        g = rng.standard_normal((n, n, n), dtype=np.float64).astype(dt)
        L = 1000.0
        kedges = np.linspace(2 * np.pi / L, np.pi * n / L, 65)
        sedges = np.linspace(L / n, 0.25 * L, 33)

        tp = timed(
            lambda: f2.power_spectrum(
                f, lbox=L, method="TSC", bins=kedges, nthreads=args.nthreads
            ),
            args.reps,
        )
        tx = timed(
            lambda: f2.correlation_function(
                f, lbox=L, method="TSC", bins=sedges, nthreads=args.nthreads
            ),
            args.reps,
        )
        ti = timed(
            lambda: f2.interlaced_power_spectrum(
                f, g, lbox=L, method="TSC", bins=kedges, nthreads=args.nthreads
            ),
            args.reps,
        )
        ts = timed(lambda: f2.smooth(f, L, 20.0, nthreads=args.nthreads), args.reps)
        tn = (
            timed(lambda: numpy_power(f, (L, L, L), kedges, 3), 1)
            if args.numpy
            else float("nan")
        )

        gb = f.nbytes / 1024**3
        print(
            f"{n:7d} {gb:8.2f}G {tp:8.2f}s {tx:8.2f}s {ti:8.2f}s {ts:8.2f}s {tn:10.2f}s {rss_gb():9.2f}G"
        )

        del f, g

    # agreement with the reference
    n = args.nmesh[0]
    f = rng.standard_normal((n, n, n), dtype=np.float64).astype(dt)
    L = 1000.0
    kedges = np.linspace(2 * np.pi / L, np.pi * n / L, 65)
    got = f2.power_spectrum(
        f, lbox=L, method="TSC", bins=kedges, nthreads=args.nthreads
    ).pk
    ref = numpy_power(f, (L, L, L), kedges, 3)
    print(
        f"\nagreement with the NumPy reference at nmesh={n}: "
        f"max relative difference {np.nanmax(np.abs(got / ref - 1)):.2e}"
    )


if __name__ == "__main__":
    main()
