# selector.py
import numpy as np
import pandas as pd
import numba as nb

@nb.njit(fastmath=True)
def compute_correlation(col1, col2):
    n = len(col1)
    mean1 = 0.0
    mean2 = 0.0
    for i in range(n):
        mean1 += col1[i]
        mean2 += col2[i]
    mean1 /= n
    mean2 /= n
    cov = 0.0
    var1 = 0.0
    var2 = 0.0
    for i in range(n):
        diff1 = col1[i] - mean1
        diff2 = col2[i] - mean2
        cov += diff1 * diff2
        var1 += diff1 * diff1
        var2 += diff2 * diff2
    if var1 == 0.0 or var2 == 0.0:
        return 0.0
    return cov / (np.sqrt(var1 * var2))

@nb.njit(fastmath=True)
def get_uncorrelated_is_numba(PLS_ordered_values, max_corr, max_columns):
    n_columns = PLS_ordered_values.shape[1]
    selected_indices = [0]
    for col in range(1, n_columns):
        is_uncorrelated = True
        for sel in selected_indices:
            corr = compute_correlation(PLS_ordered_values[:, sel], PLS_ordered_values[:, col])
            if corr >= max_corr:
                is_uncorrelated = False
                break
        if is_uncorrelated:
            selected_indices.append(col)
        if len(selected_indices) >= max_columns:
            break
    return np.array(selected_indices)

class ColumnSelector:
    def __init__(self, df, risk_free_rate=0.0):
        """
        Initialize with a DataFrame (of returns) and a risk-free rate.
        """
        self.df = df
        self.risk_free_rate = risk_free_rate
        self.data = df.values
        self.metrics_results = {}

    def select_best_columns(self, metric_func, top_n=10, metric_name="default"):
        """
        Compute the metric for each asset, sort, and select the top_n columns.
        The sort order is determined by the metric function's `ascending` attribute.
        
        Parameters:
          - metric_func: function that computes a metric for each asset.
          - top_n: number of top assets to select.
          - metric_name: a key to store the result.
        """
        data = self.data
        # Check whether the metric function expects risk_free_rate.
        if metric_func.__code__.co_argcount == 2:
            metric_values = metric_func(data, self.risk_free_rate)
        else:
            metric_values = metric_func(data)
        # Determine the sort order from the metric function attribute.
        ascending = getattr(metric_func, "ascending", False)
        if ascending:
            top_indices = np.argsort(metric_values)[:top_n]
        else:
            top_indices = np.argsort(metric_values)[::-1][:top_n]
        selected_columns = self.df.columns[top_indices]
        self.metrics_results[metric_name] = {
            "indices": top_indices,
            "values": metric_values[top_indices],
            "columns": selected_columns
        }
        return selected_columns, metric_values[top_indices]

    def filter_by_correlation(self, metric_name, max_corr=0.5, max_columns=10):
        """
        Further filter the previously selected columns so that only those assets that are pairwise
        uncorrelated (correlation < max_corr) remain.
        """
        if metric_name not in self.metrics_results:
            raise ValueError(f"No results stored for metric {metric_name}")
        indices = self.metrics_results[metric_name]["indices"]
        data_subset = self.data[:, indices]
        selected_rel_indices = get_uncorrelated_is_numba(data_subset, max_corr, max_columns)
        filtered_indices = indices[selected_rel_indices]
        filtered_columns = self.df.columns[filtered_indices]
        filtered_values = self.metrics_results[metric_name]["values"][selected_rel_indices]
        self.metrics_results[f"{metric_name}_filtered"] = {
            "indices": filtered_indices,
            "values": filtered_values,
            "columns": filtered_columns
        }
        return filtered_columns, filtered_values

    def get_metric_results(self, metric_name):
        return self.metrics_results.get(metric_name, None)
