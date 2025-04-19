# initial_selector.py
import numpy as np
import pandas as pd

class InitialSelector:
    def __init__(self, df: pd.DataFrame, risk_free_rate: float = 0.0):
        self.df = df
        self.risk_free_rate = risk_free_rate
        self.data = df.values

    def select_best(self, metric_func, top_n: int, metric_name: str = "default"):
        """
        Compute the metric for each asset and select the top_n based on the metric.
        The metric function should have an attached 'ascending' attribute.
        """
        if metric_func.__code__.co_argcount == 2:
            metric_values = metric_func(self.data, self.risk_free_rate)
        else:
            metric_values = metric_func(self.data)
        ascending = getattr(metric_func, "ascending", False)
        if ascending:
            top_indices = np.argsort(metric_values)[:top_n]
        else:
            top_indices = np.argsort(metric_values)[::-1][:top_n]
        selected_columns = self.df.columns[top_indices]
        return selected_columns, metric_values[top_indices]
