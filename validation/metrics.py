"""
The ranking objectives.

Contract, held by every entry in ``METRICS``: take a ``(T, k)`` matrix as the
first positional argument, return a ``(k,)`` vector, and carry an ``ascending``
attribute saying which end of that vector is good. Optional parameters named
``risk_free_rate``, ``lookback_bars`` or ``required_return`` are filled in by
the engine from the run configuration; anything else must have a default.

Where a metric needs a running peak or a running product, it delegates to
:mod:`validation.kernels`. Where NumPy already vectorises the work, it stays
here in NumPy — see that module's header for the measurements behind the split.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis

from .kernels import drawdown_stats


def compute_sharpe(data, risk_free_rate=0.0):
    """
    Compute the Sharpe ratio for each asset.
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    sharpe = np.zeros(data.shape[1], dtype=np.float32)
    mask = stds > 0
    sharpe[mask] = (means[mask] - risk_free_rate) / stds[mask]
    return sharpe
compute_sharpe.ascending = False  # Higher Sharpe is better


def compute_highest_return(data):
    """
    Compute the total return (sum of returns) for each asset.
    """
    return np.sum(data, axis=0)
compute_highest_return.ascending = False  # Higher return is better


def compute_max_drawdown(data, compounding: bool = True):
    """Maximum peak-to-trough decline per column, as a positive magnitude.

    The equity curve starts at 1.0 and **that starting value is already a
    peak**. The previous implementation began the running maximum at the first
    bar's close instead, so any decline from inception to the first new high
    was invisible: a column returning ``[-0.50, +1.00]`` was scored as a 0%
    drawdown rather than 50%. Strategies that begin badly are exactly the ones
    that mis-scoring flatters.
    """
    return drawdown_stats(np.ascontiguousarray(data), compounding)[0]
compute_max_drawdown.ascending = True  # Lower drawdown is better


def compute_volatility(data, annualize=False, trading_days=252):
    """
    Compute volatility (standard deviation) for each asset.
    """
    vol = np.std(data, axis=0)
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol
compute_volatility.ascending = True  # Lower volatility is better


def compute_momentum(data, lookback_bars: int = 252):
    """Sum of returns over the trailing ``lookback_bars`` **bars**.

    The unit is bars, and the name now says so. It previously read
    ``lookback`` and the engine passed it ``window_length``, which is a count
    of *months*: a 24-month in-sample window silently became a 24-*bar*
    momentum — five weeks of daily data. The engine now converts months to
    bars using the observed sampling frequency of the index.
    """
    lookback_bars = int(min(max(lookback_bars, 1), data.shape[0]))
    return np.sum(data[-lookback_bars:], axis=0)
compute_momentum.ascending = False  # Higher momentum is better


def compute_average_trade_ratio(returns, turnover, risk_free_rate=0.0):
    """
    Compute the average trade for each asset as:
       average_trade = (sum of P&L) / (sum of turnover)
    This implementation uses a true masked divide so that no /0 is ever evaluated.
    """
    # Sum up P&L and turnover per asset
    total_pl       = np.sum(returns, axis=0)
    total_turnover = np.sum(turnover, axis=0)

    # Prepare output array filled with NaN where turnover == 0
    out = np.full_like(total_pl, np.nan, dtype=float)

    # Masked division: only divide where turnover != 0
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_trade = np.divide(
            total_pl,
            total_turnover,
            out=out,
            where=(total_turnover != 0)
        )

    return avg_trade
compute_average_trade_ratio.ascending = False  # Higher average trade ratio is better



def compute_sortino(data, risk_free_rate=0.0, required_return=0.0):
    """
    Compute the Sortino ratio for each asset.
    """
    means = np.mean(data, axis=0)
    downs = np.minimum(0, data - required_return)
    dd = np.sqrt(np.mean(downs**2, axis=0))
    ratio = np.zeros(data.shape[1], dtype=np.float32)
    mask = dd > 0
    ratio[mask] = (means[mask] - required_return) / dd[mask]
    return ratio
compute_sortino.ascending = False  # Higher is better


def compute_calmar(data, compounding: bool = True):
    """Mean return per unit of maximum drawdown.

    Both terms are per period, so the ratio is scale-free in the bar frequency
    and needs no annualisation argument — the previous signature annualised the
    numerator with ``sqrt(trading_days)`` while leaving the denominator
    unscaled, which is not a Calmar ratio in any convention.
    """
    data = np.ascontiguousarray(data)
    mdd = drawdown_stats(data, compounding)[0]
    out = np.zeros(data.shape[1], dtype=np.float64)
    ok = mdd > 0
    out[ok] = np.mean(data, axis=0)[ok] / mdd[ok]
    return out
compute_calmar.ascending = False  # Higher is better


def compute_information_ratio(data, benchmark, annualize=False, trading_days=252):
    """
    Compute the Information Ratio relative to a benchmark series.
    """
    excess = data - benchmark[:, None]
    mean_exc = np.mean(excess, axis=0)
    std_exc = np.std(excess, axis=0)
    if annualize:
        mean_exc *= np.sqrt(trading_days)
        std_exc *= np.sqrt(trading_days)
    ir = np.zeros(data.shape[1], dtype=np.float32)
    mask = std_exc > 0
    ir[mask] = mean_exc[mask] / std_exc[mask]
    return ir
compute_information_ratio.ascending = False  # Higher is better


def compute_skewness(data):
    """
    Compute sample skewness for each asset.
    """
    return skew(data, axis=0)
compute_skewness.ascending = False  # More negative skew is worse


def compute_kurtosis(data):
    """
    Compute sample excess kurtosis for each asset.
    """
    return kurtosis(data, axis=0, fisher=True)
compute_kurtosis.ascending = True  # Higher kurtosis (fat tails) is worse


# Expose a dictionary mapping names to metric functions.
METRICS = {
    "sharpe": compute_sharpe,
    "highest_return": compute_highest_return,
    "max_drawdown": compute_max_drawdown,
    "volatility": compute_volatility,
    "momentum": compute_momentum,
    "avg_trade": compute_average_trade_ratio,
    "sortino": compute_sortino,
    "calmar": compute_calmar,
    "information_ratio": compute_information_ratio,
    "skewness": compute_skewness,
    "kurtosis": compute_kurtosis
}

if __name__ == "__main__":
    # Quick test on dummy data.
    np.random.seed(0)
    data = np.random.randn(252, 5).astype(np.float32) * 0.01
    print("Sharpe:", compute_sharpe(data))
    print("Avg Trade Ratio (dummy turnover):", compute_average_trade_ratio(data, np.abs(data)))
