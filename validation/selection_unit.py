"""
One in-sample window: rank, de-correlate, then (optionally) enforce a floor on
the average trade so that the shortlist cannot be filled with strategies whose
edge is smaller than the cost of trading it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .correlation_filter import CorrelationFilter
from .initial_selector import InitialSelector
from .metrics import compute_average_trade_ratio

__all__ = ["SelectionUnit"]


class SelectionUnit:
    def __init__(self, df, risk_free_rate=0.0, turnover_df=None, min_avg_trade=None,
                 use_abs_corr: bool = False):
        """
        Parameters
        ----------
        df : DataFrame-like
            In-sample returns slice. Anything exposing ``.values`` /
            ``.columns`` works, including the zero-copy ``ArrayFrame`` views.
        turnover_df : DataFrame-like, optional
            Turnover over the *same rows and columns* as ``df``.
        min_avg_trade : float, optional
            Minimum P&L per unit of turnover required to stay in the shortlist.
        """
        self.df = df
        self.risk_free_rate = risk_free_rate
        self.turnover_df = turnover_df
        self.min_avg_trade = min_avg_trade
        self.initial_selector = InitialSelector(df, risk_free_rate)
        self.correlation_filter = CorrelationFilter(df, use_abs=use_abs_corr)

    def perform_selection(self, metric_func, top_n=10, max_corr=0.5, max_columns=10,
                          metric_name="default", metric_values=None):
        selected_cols, metric_values = self.initial_selector.select_best(
            metric_func, top_n, metric_name, metric_values=metric_values
        )
        filtered_cols, filtered_values = self.correlation_filter.filter(
            selected_cols, metric_values, max_corr, max_columns
        )

        avg_trade_filtered = None
        if self.turnover_df is not None and self.min_avg_trade is not None and len(filtered_cols):
            pos = self.df.columns.get_indexer(pd.Index(filtered_cols))
            returns_data = np.asarray(self.df.values)[:, pos]
            turnover_data = np.asarray(self.turnover_df.values)[:, pos]
            avg_trade = compute_average_trade_ratio(
                returns_data, turnover_data, self.risk_free_rate
            )
            valid = np.isfinite(avg_trade) & (avg_trade >= self.min_avg_trade)
            filtered_cols = filtered_cols[valid]
            filtered_values = np.asarray(filtered_values)[valid]
            avg_trade_filtered = avg_trade[valid]

        result = {
            "selected": list(selected_cols),
            "values": list(metric_values),
            "filtered": list(filtered_cols),
            "filtered_values": list(filtered_values),
        }
        if avg_trade_filtered is not None:
            result["avg_trade"] = list(avg_trade_filtered)
        return result
