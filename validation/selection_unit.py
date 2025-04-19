# selection_unit.py
from .initial_selector import InitialSelector
from .correlation_filter import CorrelationFilter
from .metrics import compute_average_trade_ratio  # your existing function
import numpy as np

class SelectionUnit:
    def __init__(self, df, risk_free_rate=0.0, turnover_df=None, min_avg_trade=None):
        """
        Parameters:
          - df: In-sample returns DataFrame slice.
          - risk_free_rate: For metric calculations.
          - turnover_df: Optional turnover DataFrame (aligned with df).
          - min_avg_trade: Optional threshold; only assets with average trade ratio >= threshold are retained.
        """
        self.df = df
        self.risk_free_rate = risk_free_rate
        self.turnover_df = turnover_df
        self.min_avg_trade = min_avg_trade
        self.initial_selector = InitialSelector(df, risk_free_rate)
        self.correlation_filter = CorrelationFilter(df)

    def perform_selection(self, metric_func, top_n=10, max_corr=0.5, max_columns=10, metric_name="default"):
        """
        Run selection in two stages:
          1. Initial selection by ranking assets using the given metric.
          2. Filtering based on pairwise correlation.
        Optionally, further filter by average trade ratio if turnover data and a threshold are provided.
        """
        # Stage 1: Initial selection.
        selected_cols, metric_values = self.initial_selector.select_best(metric_func, top_n, metric_name)
        # Stage 2: Correlation filtering.
        filtered_cols, filtered_values = self.correlation_filter.filter(selected_cols, metric_values, max_corr, max_columns)
        
        # Additional filtering based on turnover and minimum average trade.
        if self.turnover_df is not None and self.min_avg_trade is not None:
            returns_data = self.df[filtered_cols].values
            turnover_data = self.turnover_df.loc[self.df.index, filtered_cols].values
            avg_trade = compute_average_trade_ratio(returns_data, turnover_data, self.risk_free_rate)
            valid_mask = avg_trade >= self.min_avg_trade
            filtered_cols = filtered_cols[valid_mask]
            filtered_values = filtered_values[valid_mask]
            avg_trade_filtered = avg_trade[valid_mask]
        else:
            avg_trade_filtered = None

        result = {
            "selected": list(selected_cols),
            "values": list(metric_values),
            "filtered": list(filtered_cols),
            "filtered_values": list(filtered_values)
        }
        if avg_trade_filtered is not None:
            result["avg_trade"] = list(avg_trade_filtered)
        return result
