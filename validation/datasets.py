"""
Getting from "I have a strategy" to "I have a candidate matrix".

The library takes a wide frame of candidate return streams — one column per
strategy or parameter set — which is not a thing most people have lying
around. This module closes that gap in both directions:

* :func:`candidate_matrix` turns a parameter sweep over a price series into
  exactly the ``(returns, turnover)`` pair the ensemble expects. Read it as
  the reference implementation of the input format.
* :func:`synthetic_prices` generates a realistic-looking price series with no
  edge in it, so the demo runs offline, deterministically, and with nothing to
  find — which is the point the demo is making.

No market data is shipped with this package. Point :func:`load_ohlcv` at your
own CSV to run any of this on something real.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["synthetic_prices", "load_ohlcv", "candidate_matrix", "RULES"]


# ---------------------------------------------------------------- prices
def synthetic_prices(n: int = 9000, seed: int = 7, start: str = "1990-01-01") -> pd.Series:
    """A price series with the texture of a market and none of the signal.

    Volatility clusters (a simple GARCH-ish recursion) and returns are
    fat-tailed, so trend rules behave the way they do on real data: long flat
    stretches, occasional violent moves. The drift is a constant, and there is
    nothing in the path that a look-back rule can predict.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)

    vol = np.empty(n)
    vol[0] = 0.010
    shock = rng.standard_t(df=4.0, size=n) / np.sqrt(4.0 / 2.0)
    ret = np.empty(n)
    for i in range(n):
        if i:
            # vol today = base + persistence * vol yesterday + reaction to |shock|
            vol[i] = 0.0016 + 0.86 * vol[i - 1] + 0.10 * abs(ret[i - 1])
        ret[i] = 0.00022 + vol[i] * shock[i]

    return pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx, name="CLOSE")


def load_ohlcv(path, close_col: Optional[str] = None, date_col: Optional[str] = None
               ) -> pd.Series:
    """Read a close series out of a CSV. Column names are sniffed, not assumed."""
    df = pd.read_csv(path)
    if date_col is None:
        for c in df.columns:
            if str(c).lower() in ("date", "data", "datetime", "timestamp", "time"):
                date_col = c
                break
        else:
            date_col = df.columns[0]
    if close_col is None:
        for c in df.columns:
            if str(c).lower() in ("close", "adj close", "adj_close", "chiusura"):
                close_col = c
                break
        else:
            raise ValueError(f"no close column found in {list(df.columns)}")

    s = pd.Series(
        pd.to_numeric(df[close_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col], errors="coerce"),
        name="CLOSE",
    )
    return s[s.index.notna() & s.notna()].sort_index()


# ---------------------------------------------------------------- rules
def _breakout(px: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Donchian: long above the rolling high, short below the rolling low."""
    s = pd.Series(px)
    hi = s.rolling(slow).max().shift(1).to_numpy()
    lo = s.rolling(slow).min().shift(1).to_numpy()
    mid = s.rolling(fast).mean().shift(1).to_numpy()
    pos = np.zeros(len(px))
    pos[px > hi] = 1.0
    pos[px < lo] = -1.0
    pos = pd.Series(pos).replace(0.0, np.nan).ffill().fillna(0.0).to_numpy().copy()
    # flatten when price crosses back through the fast mean
    ok = np.isfinite(mid)
    pos[ok & (pos > 0) & (px < mid)] = 0.0
    pos[ok & (pos < 0) & (px > mid)] = 0.0
    return pos


def _crossover(px: np.ndarray, fast: int, slow: int) -> np.ndarray:
    s = pd.Series(px)
    f = s.ewm(span=fast, adjust=False).mean().shift(1).to_numpy()
    sl = s.ewm(span=slow, adjust=False).mean().shift(1).to_numpy()
    return np.where(f > sl, 1.0, -1.0)


def _tsmom(px: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """Time-series momentum on the sign of the trailing `slow`-day return."""
    s = pd.Series(px)
    past = s.shift(slow).to_numpy()
    raw = np.where(px > past, 1.0, -1.0)
    raw[~np.isfinite(past)] = 0.0
    return pd.Series(raw).rolling(fast, min_periods=1).mean().to_numpy()


RULES = {"breakout": _breakout, "crossover": _crossover, "tsmom": _tsmom}


# ---------------------------------------------------------------- matrix
def candidate_matrix(
    prices: pd.Series,
    rules: Sequence[str] = ("breakout", "crossover", "tsmom"),
    fasts: Iterable[int] = (5, 10, 15, 20, 30, 40),
    slows: Iterable[int] = (40, 60, 80, 120, 160, 250),
    cost_bps: float = 2.0,
    vol_target: Optional[float] = 0.10,
    vol_window: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep a parameter grid over one price series into ``(returns, turnover)``.

    Every column is one strategy. Everything is causal: the position at bar
    ``t`` is decided from data up to ``t-1`` and earns the return of bar ``t``.
    Cost is charged on the change in position, which is also what turnover
    measures — so ``sum(returns) / sum(turnover)`` is the average trade, and
    the ``min_avg_trade`` gate has something meaningful to filter on.

    Returns
    -------
    (returns, turnover) : two DataFrames, same index, same columns.
        Exactly the pair :class:`FullBacktesterEnsemble` expects.
    """
    px = prices.to_numpy(dtype=np.float64)
    idx = prices.index
    ret1 = np.zeros(len(px))
    ret1[1:] = px[1:] / px[:-1] - 1.0

    scale = np.ones(len(px))
    if vol_target:
        rolling_vol = pd.Series(ret1).rolling(vol_window).std().shift(1).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = vol_target / (rolling_vol * np.sqrt(252.0))
        scale = np.clip(np.nan_to_num(scale, nan=0.0, posinf=0.0), 0.0, 3.0)

    cost = cost_bps * 1e-4
    cols_r, cols_t, names = [], [], []

    for rule in rules:
        fn = RULES[rule]
        for fast in fasts:
            for slow in slows:
                if fast >= slow:
                    continue
                raw = fn(px, int(fast), int(slow))
                pos = np.nan_to_num(raw, nan=0.0) * scale
                pos = np.concatenate(([0.0], pos[:-1]))     # execute next bar
                trade = np.abs(np.diff(pos, prepend=0.0))
                cols_r.append(pos * ret1 - cost * trade)
                cols_t.append(trade)
                names.append(f"{rule}_f{fast}_s{slow}")

    returns = pd.DataFrame(np.column_stack(cols_r), index=idx, columns=names)
    turnover = pd.DataFrame(np.column_stack(cols_t), index=idx, columns=names)
    return returns, turnover
