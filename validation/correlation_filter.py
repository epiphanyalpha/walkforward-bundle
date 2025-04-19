# correlation_filter.py
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
def get_uncorrelated_indices(selected_data, max_corr, max_columns):
    n_columns = selected_data.shape[1]
    selected_indices = [0]
    for col in range(1, n_columns):
        is_uncorrelated = True
        for sel in selected_indices:
            corr = compute_correlation(selected_data[:, sel], selected_data[:, col])
            if corr >= max_corr:
                is_uncorrelated = False
                break
        if is_uncorrelated:
            selected_indices.append(col)
        if len(selected_indices) >= max_columns:
            break
    return np.array(selected_indices)

class CorrelationFilter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.data = df.values

    def filter(self, selected_columns: pd.Index, metric_values: np.ndarray, max_corr: float, max_columns: int):
        """
        From the initially selected columns, remove assets that are too correlated.
        """
        indices = self.df.columns.get_indexer(selected_columns)
        selected_data = self.data[:, indices]
        filtered_rel_indices = get_uncorrelated_indices(selected_data, max_corr, max_columns)
        filtered_indices = indices[filtered_rel_indices]
        filtered_columns = self.df.columns[filtered_indices]
        filtered_metric_values = metric_values[filtered_rel_indices]
        return filtered_columns, filtered_metric_values
