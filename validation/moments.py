"""
Metric evaluation that stops recomputing what it already knows.

Two observations drive this module.

**One.** Sharpe, volatility and total return over a window are functions of the
first two moments, and moments are additive.  Cache the prefix sums of ``x``
and ``x**2`` once — ``O(T*k)`` — and every window afterwards costs ``O(k)``,
one subtraction per candidate, no matter how long the window is.  On an
expanding walk-forward, where late windows span nearly the whole history, this
turns the dominant cost of the run into a rounding error.

**Two.** Across a configuration grid, many configs share the *same* in-sample
windows: ``max_corr`` and ``max_columns`` change what happens *after* the
ranking, not the ranking itself.  A cache keyed on
``(metric, window_start, window_end)`` therefore collapses the whole
``max_corr x max_columns`` face of the grid into a single evaluation.

Path-dependent metrics — drawdown, Calmar, Sortino, skew — are not moments and
get the ordinary treatment: computed directly, but still cached per window.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = ["MomentCache", "MetricEngine", "REDUCIBLE_METRICS"]

# Metrics expressible from prefix sums of x and x**2 over the window.
REDUCIBLE_METRICS = frozenset({"sharpe", "volatility", "highest_return", "momentum"})


class MomentCache:
    """Prefix sums of ``x`` and ``x**2`` down the time axis.

    Memory cost is ``2 * (T+1) * k * 8`` bytes — for a 5,000 x 1,000 universe
    that is ~80 MB, roughly twice the float32 matrix itself.  It buys back an
    ``O(T)`` factor on every window, so on anything but a trivially short
    history it is the right trade; if it is not, pass ``moments=False``.
    """

    __slots__ = ("n", "k", "_cs", "_cs2")

    def __init__(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        self.n, self.k = x.shape
        # float64 accumulation: differencing two large prefix sums is where
        # precision goes to die, and float32 does not have enough of it.
        self._cs = np.zeros((self.n + 1, self.k), dtype=np.float64)
        self._cs2 = np.zeros((self.n + 1, self.k), dtype=np.float64)
        np.cumsum(x, axis=0, out=self._cs[1:])
        np.cumsum(x * x, axis=0, out=self._cs2[1:])

    @property
    def nbytes(self) -> int:
        return self._cs.nbytes + self._cs2.nbytes

    def sum(self, i0: int, i1: int) -> np.ndarray:
        return self._cs[i1] - self._cs[i0]

    def sum_sq(self, i0: int, i1: int) -> np.ndarray:
        return self._cs2[i1] - self._cs2[i0]

    def mean_var(self, i0: int, i1: int) -> Tuple[np.ndarray, np.ndarray]:
        """Population mean and variance over rows ``[i0, i1)``."""
        n = i1 - i0
        if n <= 0:
            z = np.zeros(self.k)
            return z, z
        s = self.sum(i0, i1)
        s2 = self.sum_sq(i0, i1)
        mean = s / n
        var = s2 / n - mean * mean
        # tiny negatives are cancellation noise, not negative variance
        np.maximum(var, 0.0, out=var)
        return mean, var


class MetricEngine:
    """Per-window metric values, computed the cheap way and remembered.

    The cache is process-local and lives as long as the bundle does.  Under
    ``n_jobs > 1`` each worker builds its own — the sharing happens *within* a
    process, so a serial run over a wide grid benefits most.
    """

    def __init__(self, bundle, moments: bool = True) -> None:
        self._bundle = bundle
        self._use_moments = moments
        self._mc: Optional[MomentCache] = None
        self._cache: Dict[Tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    @property
    def moment_cache(self) -> MomentCache:
        if self._mc is None:
            self._mc = MomentCache(self._bundle.returns)
        return self._mc

    def clear(self) -> None:
        self._cache.clear()

    def _closed_form(self, name: str, i0: int, i1: int, risk_free_rate: float,
                     lookback: int) -> Optional[np.ndarray]:
        n = i1 - i0
        if n <= 0:
            return None
        mc = self.moment_cache

        if name == "highest_return":
            return mc.sum(i0, i1)

        if name == "momentum":
            # sum of the last `lookback` rows of the window
            lb = min(int(lookback), n)
            return mc.sum(i1 - lb, i1)

        mean, var = mc.mean_var(i0, i1)
        std = np.sqrt(var)

        if name == "volatility":
            return std

        if name == "sharpe":
            out = np.zeros(self._bundle.returns.shape[1], dtype=np.float64)
            ok = std > 0
            out[ok] = (mean[ok] - risk_free_rate) / std[ok]
            return out

        return None

    # ------------------------------------------------------------------
    def compute(self, name: str, metric_func: Callable, i0: int, i1: int,
                risk_free_rate: float = 0.0, lookback: int = 252) -> np.ndarray:
        key = (name, i0, i1, float(risk_free_rate), int(lookback))
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1

        values = None
        if self._use_moments and name in REDUCIBLE_METRICS:
            values = self._closed_form(name, i0, i1, risk_free_rate, lookback)

        if values is None:
            import inspect

            kwargs = {}
            try:
                params = inspect.signature(metric_func).parameters
            except (TypeError, ValueError):
                params = {}
            for p in params:
                if p == "risk_free_rate":
                    kwargs["risk_free_rate"] = risk_free_rate
                elif p in ("lookback_bars", "lookback", "momentum_lookback"):
                    kwargs[p] = lookback
            values = np.asarray(metric_func(self._bundle.returns[i0:i1], **kwargs))

        values = np.ascontiguousarray(values)
        values.flags.writeable = False
        self._cache[key] = values
        return values

    def stats(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._cache),
            "moment_cache_bytes": self._mc.nbytes if self._mc is not None else 0,
        }
