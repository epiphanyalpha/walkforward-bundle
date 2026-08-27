"""
Stage 1 of the selection: rank every candidate by an in-sample metric.
"""
from __future__ import annotations

import inspect

import numpy as np

__all__ = ["InitialSelector"]


def _rank_top_n(values: np.ndarray, top_n: int, ascending: bool) -> np.ndarray:
    """Indices of the ``top_n`` best entries, NaN-safe.

    ``np.argsort`` parks NaNs at the end; reversing it for a descending sort
    therefore puts them *first*, and a candidate with an undefined metric wins
    the slot.  Pushing non-finite values to the losing extreme before sorting
    is the fix.  ``+inf`` is demoted too: an infinite Sharpe or Calmar means a
    zero-variance / zero-drawdown candidate, which is a data artefact rather
    than the best strategy in the universe.

    Uses ``argpartition`` (O(k)) rather than a full sort (O(k log k)); on a
    universe of a few thousand candidates, ranked once per window per config,
    that difference is not free.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    bad = ~np.isfinite(v)
    if bad.any():
        v = v.copy()
        v[bad] = np.inf if ascending else -np.inf

    top_n = int(min(max(top_n, 0), v.size))
    if top_n == 0:
        return np.empty(0, dtype=np.int64)

    if ascending:
        idx = np.argpartition(v, top_n - 1)[:top_n]
        return idx[np.argsort(v[idx], kind="stable")]
    idx = np.argpartition(-v, top_n - 1)[:top_n]
    return idx[np.argsort(-v[idx], kind="stable")]


class InitialSelector:
    """Rank candidates by ``metric_func`` and keep the best ``top_n``.

    ``metric_func`` is called with the raw ``(T, k)`` matrix.  Its signature is
    inspected so that optional arguments (``risk_free_rate``, ``lookback``,
    ``momentum_lookback``) are filled in automatically, and the direction of
    "better" is read from the function's ``ascending`` attribute.
    """

    def __init__(self, df, risk_free_rate: float = 0.0, window_length: int = 12):
        self.df = df
        self.risk_free_rate = risk_free_rate
        self.window_length = window_length
        self.data = np.asarray(df.values)

    def select_best(self, metric_func, top_n: int, metric_name: str = "default",
                    metric_values=None):
        if metric_values is not None:
            metric_values = np.asarray(metric_values)
            ascending = bool(getattr(metric_func, "ascending", False))
            top_indices = _rank_top_n(metric_values, top_n, ascending)
            return self.df.columns[top_indices], metric_values[top_indices]

        kwargs = {}
        try:
            params = inspect.signature(metric_func).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins
            params = {}
        for name in params:
            if name == "risk_free_rate":
                kwargs["risk_free_rate"] = self.risk_free_rate
            elif name in ("lookback", "momentum_lookback"):
                kwargs[name] = self.window_length

        metric_values = np.asarray(metric_func(self.data, **kwargs))
        ascending = bool(getattr(metric_func, "ascending", False))
        top_indices = _rank_top_n(metric_values, top_n, ascending)
        return self.df.columns[top_indices], metric_values[top_indices]
