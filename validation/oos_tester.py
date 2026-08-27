"""
The out-of-sample leg: hold the selection fixed, measure what it did next.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kernels import row_mean_subset

__all__ = [
    "compute_cumulative_return",
    "compute_oos_volatility",
    "compute_oos_sharpe",
    "compute_portfolio_avg_trade",
    "OutOfSampleTester",
]


def compute_cumulative_return(data):
    return np.prod(1 + np.asarray(data, dtype=np.float64)) - 1


def compute_oos_volatility(data, annualize=False, periods_per_year=252):
    vol = float(np.std(np.asarray(data, dtype=np.float64)))
    if annualize:
        vol *= np.sqrt(periods_per_year)
    return vol


def compute_oos_sharpe(data, risk_free_rate=0.0, annualize=False, periods_per_year=252):
    d = np.asarray(data, dtype=np.float64)
    std_val = float(np.std(d))
    if std_val == 0.0 or not np.isfinite(std_val):
        return np.nan
    sr = (float(np.mean(d)) - risk_free_rate) / std_val
    return sr * np.sqrt(periods_per_year) if annualize else sr


def compute_portfolio_avg_trade(portfolio_returns, portfolio_turnover):
    """P&L per unit of turnover — the edge per trade, in return units."""
    total_pl = float(np.sum(portfolio_returns))
    total_turnover = float(np.sum(portfolio_turnover))
    return total_pl / total_turnover if total_turnover != 0 else np.nan


class OutOfSampleTester:
    """Equal-weight the selected columns over one out-of-sample period."""

    def __init__(self, oos_df, selected_columns, turnover_oos_df=None, risk_free_rate=0.0,
                 periods_per_year=252):
        self.oos_df = oos_df
        self.selected_columns = pd.Index(selected_columns)
        self.turnover_oos_df = turnover_oos_df
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    @staticmethod
    def _equal_weight(frame, pos: np.ndarray) -> np.ndarray:
        """Equal-weight row mean over ``pos``, without copying the submatrix."""
        values = np.asarray(frame.values)
        if not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)
        return row_mean_subset(values, np.ascontiguousarray(pos, dtype=np.int64))

    def run(self):
        pos = self.oos_df.columns.get_indexer(self.selected_columns)
        pos = pos[pos >= 0]
        if pos.size == 0:
            return {}

        index = self.oos_df.index
        portfolio_returns = pd.Series(self._equal_weight(self.oos_df, pos), index=index)
        result = {
            "portfolio_returns_series": portfolio_returns,
            "cumulative_return": compute_cumulative_return(portfolio_returns.values),
            "oos_volatility": compute_oos_volatility(
                portfolio_returns.values, annualize=True,
                periods_per_year=self.periods_per_year),
            "oos_sharpe": compute_oos_sharpe(
                portfolio_returns.values, self.risk_free_rate),
        }
        if self.turnover_oos_df is not None:
            portfolio_turnover = pd.Series(
                self._equal_weight(self.turnover_oos_df, pos), index=index
            )
            result["portfolio_turnover_series"] = portfolio_turnover
            result["portfolio_avg_trade"] = compute_portfolio_avg_trade(
                portfolio_returns.values, portfolio_turnover.values
            )
        return result
