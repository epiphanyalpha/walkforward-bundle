"""
Compiled column kernels.

Every function here takes one contiguous ``(T, k)`` float64 matrix of candidate
returns and produces one ``(k,)`` float64 vector. No pandas, no configuration,
no state: a kernel is a pure function of its matrix and its scalars, which is
what makes it safe to compile, to cache, and to run across threads.

Why they exist. The obvious NumPy phrasing of a drawdown is::

    cum      = np.cumprod(1 + data, axis=0)      # (T, k) temporary
    peak     = np.maximum.accumulate(cum, 0)     # (T, k) temporary
    drawdown = (cum - peak) / peak               # (T, k) temporary

Three full-size allocations and three passes over memory to produce ``k``
numbers. Profiling a configuration sweep put 56% of total runtime in exactly
those two accumulate calls. The kernels below carry the running product and the
running peak in registers, one column at a time, and allocate nothing but the
``(k,)`` result — one pass, no temporaries.

What is NOT here, and why. Every candidate kernel was benchmarked against its
NumPy phrasing on a 4000x800 matrix before being kept. Numba wins decisively
where a *sequential dependency* blocks vectorisation --- a running peak, a
running product --- and loses where NumPy already dispatches to a SIMD
reduction:

===========================  ==========  =========  ==============
operation                    NumPy       Numba      kept
===========================  ==========  =========  ==============
drawdown + ulcer               101.2 ms     7.1 ms  numba, 14.3x
row mean over a subset          148 us       28 us  numba, 5.3x
standardize columns             432 us      303 us  numba, 1.4x
column mean + sd                 8.9 ms    10.5 ms  NumPy
downside deviation               4.9 ms     8.2 ms  NumPy
===========================  ==========  =========  ==============

So ``mean``, ``sd`` and downside deviation are deliberately absent: writing
them here would have been decoration, and slower decoration at that. The rule
this module follows is that a kernel earns its compilation by a measurement,
not by the fact that a loop can be written.

Threading. ``drawdown_stats`` carries independent columns and uses ``prange``;
on the shortlist-sized inputs the other kernels see, thread setup costs more
than it saves, so they stay serial. Numba honours ``NUMBA_NUM_THREADS`` --- set
it to 1 when running the ensemble under a multi-process ``n_jobs``, to avoid
oversubscribing the machine.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    from numba import prange

    NUMBA_OK = True
except Exception:  # pragma: no cover - numba is a declared dependency
    NUMBA_OK = False
    prange = range

__all__ = [
    "NUMBA_OK",
    "drawdown_stats",
    "standardize",
    "greedy_uncorrelated",
    "row_mean_subset",
    "warmup",
]

_EPS = 1e-12


def _kernel(**kw):
    """Compile if numba is present, otherwise fall back to the Python body."""
    def deco(fn):
        return nb.njit(**kw)(fn) if NUMBA_OK else fn
    return deco


# --------------------------------------------------------- drawdown
@_kernel(cache=True, fastmath=True, parallel=True, nogil=True)
def drawdown_stats(data, compounding):
    """Max drawdown and ulcer index per column, one pass, no temporaries.

    ``compounding=True`` tracks the equity curve as a running product of
    ``(1 + r)``; ``False`` tracks it as a running sum, which is the right
    reading when the columns are P&L at fixed notional rather than returns on
    a reinvested balance.

    Returns drawdowns as positive magnitudes, so 0.30 means a 30% peak-to-
    trough decline. A column that never declines returns 0.0.
    """
    n, k = data.shape
    max_dd = np.zeros(k, dtype=np.float64)
    ulcer = np.zeros(k, dtype=np.float64)
    # Accumulators below are float64 literals, so a float32 input promotes on
    # every operation: the caller keeps its compact storage and the running
    # product still carries full precision. Numba compiles one specialisation
    # per input dtype, which is why no conversion happens at the boundary.
    for j in prange(k):
        equity = 1.0 if compounding else 0.0
        peak = equity
        worst = 0.0
        sq = 0.0
        for i in range(n):
            if compounding:
                equity *= 1.0 + data[i, j]
            else:
                equity += data[i, j]
            if equity > peak:
                peak = equity
            denom = peak if compounding else 1.0
            if denom > _EPS or not compounding:
                dd = (peak - equity) / denom if compounding else (peak - equity)
                if dd > worst:
                    worst = dd
                sq += dd * dd
        max_dd[j] = worst
        ulcer[j] = np.sqrt(sq / n) if n > 0 else 0.0
    return max_dd, ulcer


# ---------------------------------------------------------- correlation
@_kernel(cache=True, fastmath=True, nogil=True)
def standardize(data):
    """Centre each column and scale it to unit L2 norm.

    After this transform a Pearson correlation is a plain dot product, so the
    greedy filter below costs one multiply-add per row per comparison — no
    means, no variances, no divisions in the inner loop.

    A constant column becomes all-zero, which makes every correlation against
    it exactly 0.0. That matches the ``variance == 0 -> 0.0`` convention of the
    pairwise implementation it replaced.
    """
    n, k = data.shape
    out = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        m = 0.0
        for i in range(n):
            m += data[i, j]
        m /= n
        ss = 0.0
        for i in range(n):
            d = data[i, j] - m
            out[i, j] = d
            ss += d * d
        if ss > 0.0:
            inv = 1.0 / np.sqrt(ss)
            for i in range(n):
                out[i, j] *= inv
    return out


@_kernel(cache=True, fastmath=True, nogil=True)
def greedy_uncorrelated(z, max_corr, max_columns, use_abs):
    """Admit columns best-first while pairwise correlation stays below the cap.

    ``z`` must already be standardized and already ordered best-first: the
    first column is admitted unconditionally, and the result is therefore a
    valid selection rather than an optimal one. Deliberate — a greedy,
    deterministic, explicable rule is worth more than an optimal one when the
    output is a robustness claim.

    Sequential by nature (each decision depends on every previous admission),
    so no ``prange`` here.
    """
    n, k = z.shape
    keep = np.empty(k, dtype=np.int64)
    if k == 0 or max_columns <= 0:
        return keep[:0]

    keep[0] = 0
    n_keep = 1
    for col in range(1, k):
        if n_keep >= max_columns:
            break
        admit = True
        for s in range(n_keep):
            sel = keep[s]
            c = 0.0
            for i in range(n):
                c += z[i, sel] * z[i, col]
            if use_abs and c < 0.0:
                c = -c
            if c >= max_corr:
                admit = False
                break
        if admit:
            keep[n_keep] = col
            n_keep += 1
    return keep[:n_keep]


# --------------------------------------------------------------- subset
@_kernel(cache=True, fastmath=True, nogil=True)
def row_mean_subset(data, cols):
    """Equal-weight row mean over a column subset, without materialising it.

    ``data[:, cols].mean(axis=1)`` builds a ``(T, |cols|)`` copy first. With one
    call per out-of-sample leg per configuration that copy is pure waste; this
    reads the strided columns in place, accumulating in float64 whatever the
    storage dtype.
    """
    n = data.shape[0]
    m = cols.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if m == 0:
        return out
    inv = 1.0 / m
    for i in range(n):
        acc = 0.0
        for c in range(m):
            acc += data[i, cols[c]]
        out[i] = acc * inv
    return out


def warmup() -> None:
    """Pay the JIT cost up front, on tiny inputs, instead of mid-benchmark."""
    if not NUMBA_OK:
        return
    base = np.random.default_rng(0).normal(0, 0.01, (8, 3))
    cols = np.array([0, 2], dtype=np.int64)
    for dtype in (np.float32, np.float64):      # one specialisation per dtype
        a = np.ascontiguousarray(base, dtype=dtype)
        drawdown_stats(a, True)
        drawdown_stats(a, False)
        z = standardize(a)
        greedy_uncorrelated(z, 0.5, 2, False)
        row_mean_subset(a, cols)
