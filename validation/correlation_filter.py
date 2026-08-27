"""
Greedy de-correlation of the in-sample shortlist.

Naive implementation: for every candidate pair, walk both columns twice
(means, then moments).  With a shortlist of ``k`` candidates that is
``O(k^2 * T)`` full passes, and it recomputes the mean and variance of the
*same* column dozens of times.

Here every column is standardized once — ``(x - mean) / ||x - mean||`` — after
which a Pearson correlation is a single dot product.  Same numbers (to fp
tolerance), one pass over the data instead of ``k`` of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kernels import NUMBA_OK, greedy_uncorrelated as _greedy, standardize

__all__ = [
    "standardize_columns",
    "get_uncorrelated_indices",
    "CorrelationFilter",
    "NUMBA_OK",
]

#: Retained under its historical name; the implementation is the shared kernel.
standardize_columns = standardize


def get_uncorrelated_indices(selected_data, max_corr, max_columns, use_abs=False):
    """Greedy de-correlation over the columns of ``selected_data``.

    Columns are assumed pre-sorted best-first: the first is always kept, and
    each subsequent one survives only if its correlation with every kept
    column is below ``max_corr``.

    ``use_abs=False`` keeps the original semantics, under which a strongly
    *negative* correlation counts as diversifying and is accepted. Pass
    ``use_abs=True`` to reject on magnitude instead.
    """
    data = np.ascontiguousarray(selected_data)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        return np.empty(0, dtype=np.int64)
    return _greedy(standardize(data), float(max_corr), int(max_columns), bool(use_abs))


class CorrelationFilter:
    """Stage 2 of the selection: drop shortlisted candidates that duplicate risk."""

    def __init__(self, df, use_abs: bool = False):
        self.df = df
        self.data = np.asarray(df.values)
        self.use_abs = use_abs

    def filter(self, selected_columns, metric_values, max_corr, max_columns):
        columns = self.df.columns
        indices = columns.get_indexer(pd.Index(selected_columns))
        indices = indices[indices >= 0]
        if indices.size == 0:
            return columns[:0], np.asarray(metric_values)[:0]

        selected_data = self.data[:, indices]
        rel = get_uncorrelated_indices(
            selected_data, max_corr, max_columns, use_abs=self.use_abs
        )
        filtered_indices = indices[rel]
        return columns[filtered_indices], np.asarray(metric_values)[rel]
