"""
The walk-forward schedule and the in-sample pass over it.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from ._frames import FrameBundle
from .selection_unit import SelectionUnit

__all__ = ["WalkForwardSchedule", "WalkForwardRunner"]


class WalkForwardSchedule:
    """The calendar of in-sample windows.

    ``first_os`` is the boundary between the first in-sample window and the
    first out-of-sample period, so the first window spans
    ``[first_os - window_length, first_os]``.  Each subsequent window ends
    ``step_months`` later; ``step_months`` therefore doubles as the length of
    the out-of-sample period, which is what makes the OOS legs contiguous.

    anchored=True   expanding window, start pinned at the first date
    anchored=False  rolling window of fixed length
    """

    def __init__(self, df, first_os, window_length, anchored=True, step_months=12):
        index = df.index if hasattr(df, "index") else df
        self.index = index
        self.first_os = pd.to_datetime(first_os)
        self.window_length = int(window_length)
        self.anchored = bool(anchored)
        self.step_months = int(step_months)
        self.slices = self._generate_slices()

    def _generate_slices(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        analysis_start = self.first_os - pd.DateOffset(months=self.window_length)
        slices: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        # Stop one step short of the end: a window with no OOS period to
        # follow it would contribute an in-sample selection and nothing to
        # judge it by.
        max_allowed_end = self.index.max() - pd.DateOffset(months=self.step_months)

        if self.anchored:
            slices.append((analysis_start, self.first_os))
            current_end = self.first_os + pd.DateOffset(months=self.step_months)
            while current_end <= max_allowed_end:
                slices.append((analysis_start, current_end))
                current_end += pd.DateOffset(months=self.step_months)
        else:
            window_offset = pd.DateOffset(months=self.window_length)
            current_start = analysis_start
            current_end = current_start + window_offset
            while current_end <= max_allowed_end:
                slices.append((current_start, current_end))
                current_start += pd.DateOffset(months=self.step_months)
                current_end = current_start + window_offset
        return slices

    def get_slices(self):
        return self.slices

    def __len__(self) -> int:
        return len(self.slices)


class WalkForwardRunner:
    """Run the in-sample selection over every window of a schedule.

    Accepts either a plain DataFrame or a pre-built :class:`FrameBundle`.
    Passing the bundle is what you want inside an ensemble sweep: the matrices
    are materialized once and every window is a view into them.
    """

    def __init__(self, df, schedule, risk_free_rate=0.0, metric_func=None,
                 top_n=10, max_corr=0.5, max_columns=10, turnover_df=None,
                 min_avg_trade=None, use_abs_corr=False, metric_name=None,
                 metric_lookback=12, use_metric_cache=True):
        if isinstance(df, FrameBundle):
            self.bundle = df
            if turnover_df is None:
                self.bundle_has_turnover = self.bundle.turnover is not None
            else:
                self.bundle_has_turnover = True
        else:
            self.bundle = FrameBundle(df, turnover_df)
            self.bundle_has_turnover = turnover_df is not None

        self.use_turnover = self.bundle.turnover is not None and turnover_df is not False
        self.schedule = schedule
        self.risk_free_rate = risk_free_rate
        self.metric_func = metric_func
        self.top_n = top_n
        self.max_corr = max_corr
        self.max_columns = max_columns
        self.min_avg_trade = min_avg_trade
        self.use_abs_corr = use_abs_corr
        self.metric_name = metric_name or getattr(metric_func, "__name__", "metric")
        self.metric_lookback = metric_lookback
        self.use_metric_cache = use_metric_cache
        self.results = {}

    def run(self):
        for start, end in self.schedule.get_slices():
            i0, i1 = self.bundle.bounds(start, end)
            if i1 <= i0:
                continue
            ret_slice, tur_slice = self.bundle.islice(i0, i1)

            metric_values = None
            if self.use_metric_cache and self.metric_func is not None:
                metric_values = self.bundle.metrics.compute(
                    self.metric_name, self.metric_func, i0, i1,
                    risk_free_rate=self.risk_free_rate,
                    lookback=self.metric_lookback,
                )

            su = SelectionUnit(
                ret_slice,
                self.risk_free_rate,
                turnover_df=tur_slice if self.use_turnover else None,
                min_avg_trade=self.min_avg_trade,
                use_abs_corr=self.use_abs_corr,
            )
            result = su.perform_selection(
                self.metric_func, self.top_n, self.max_corr, self.max_columns,
                metric_name=self.metric_name, metric_values=metric_values,
            )
            self.results[(pd.Timestamp(start), pd.Timestamp(end))] = result
        return self.results
