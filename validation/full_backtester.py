"""
One complete walk-forward run: schedule -> in-sample selection -> out-of-sample
evaluation -> a single stitched OOS return stream.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._frames import FrameBundle
from .oos_tester import OutOfSampleTester
from .walkforward import WalkForwardRunner, WalkForwardSchedule

__all__ = ["FullBacktester"]


class FullBacktester:
    """A single configuration of the walk-forward.

    One instance == one point in the configuration grid.  The ensemble is what
    you get by running many of these and looking at the distribution rather
    than at any one of them.
    """

    def __init__(self, df, turnover_df, first_os, window_length, step_months=12,
                 anchored=True, risk_free_rate=0.0, metric_func=None, top_n=10,
                 max_corr=0.5, max_columns=10, min_avg_trade=None,
                 include_turnover=True, periods_per_year=252, use_abs_corr=False,
                 metric_name=None, use_metric_cache=True, momentum_months=12):
        if isinstance(df, FrameBundle):
            self.bundle = df
        else:
            self.bundle = FrameBundle(df, turnover_df)

        self.first_os = first_os
        self.window_length = window_length
        self.step_months = step_months
        self.anchored = anchored
        self.risk_free_rate = risk_free_rate
        self.metric_func = metric_func
        self.top_n = top_n
        self.max_corr = max_corr
        self.max_columns = max_columns
        self.min_avg_trade = min_avg_trade
        self.include_turnover = include_turnover and self.bundle.turnover is not None
        self.periods_per_year = periods_per_year
        self.use_abs_corr = use_abs_corr
        self.metric_name = metric_name
        self.use_metric_cache = use_metric_cache
        self.momentum_months = momentum_months

        self.schedule = WalkForwardSchedule(
            self.bundle, first_os, window_length, anchored=anchored, step_months=step_months
        )
        self.in_sample_results = {}
        self.oos_results = {}

    # ------------------------------------------------------------------
    def run_in_sample(self):
        runner = WalkForwardRunner(
            self.bundle,
            self.schedule,
            risk_free_rate=self.risk_free_rate,
            metric_func=self.metric_func,
            top_n=self.top_n,
            max_corr=self.max_corr,
            max_columns=self.max_columns,
            turnover_df=None if self.include_turnover else False,
            min_avg_trade=self.min_avg_trade,
            use_abs_corr=self.use_abs_corr,
            metric_name=self.metric_name,
            # months in, bars out: the config speaks calendar, kernels speak bars
            metric_lookback=self.bundle.bars_for_months(self.momentum_months),
            use_metric_cache=self.use_metric_cache,
        )
        runner.use_turnover = self.include_turnover
        self.in_sample_results = runner.run()
        return self.in_sample_results

    def run_oos(self):
        """Evaluate each selection on the period that follows its window.

        The OOS period starts one day after the in-sample window closes, so
        the two never share a bar — the selection is made strictly on data
        that precedes what it is judged on.
        """
        for (start, insample_end), sel in self.in_sample_results.items():
            selected = sel.get("filtered") or []
            if len(selected) == 0:
                continue
            oos_start = insample_end + pd.Timedelta(days=1)
            oos_end = insample_end + pd.DateOffset(months=self.step_months)
            oos_slice, oos_turnover = self.bundle.slice(oos_start, oos_end)
            if oos_slice.empty:
                continue
            tester = OutOfSampleTester(
                oos_slice,
                selected,
                turnover_oos_df=oos_turnover if self.include_turnover else None,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year,
            )
            res = tester.run()
            if res:
                self.oos_results[(start, insample_end)] = res
        return self.oos_results

    def aggregate_oos(self) -> Optional[dict]:
        """Stitch the per-period OOS legs into one continuous track record."""
        returns = [
            r["portfolio_returns_series"]
            for r in self.oos_results.values()
            if r.get("portfolio_returns_series") is not None
        ]
        if not returns:
            return None

        full_returns = pd.concat(returns).sort_index()
        full_returns = full_returns[~full_returns.index.duplicated(keep="first")]
        vals = full_returns.to_numpy(dtype=np.float64)

        vol = float(np.std(vals))
        # Two return figures, named for their convention rather than left to
        # be reconciled by the reader: the Sharpe below is built from simple
        # per-bar returns (additive), while an investor who reinvests realises
        # the compounded figure. Both are correct; only reporting them under
        # one ambiguous name is not.
        result = {
            "full_oos_series": full_returns,
            "overall_return_compounded": float(np.prod(1 + vals) - 1),
            "overall_return_additive": float(vals.sum()),
            "overall_volatility": vol,
            "overall_sharpe": (float(np.mean(vals)) - self.risk_free_rate) / vol
            if vol != 0 else np.nan,
            "n_periods": len(self.oos_results),
        }

        if self.include_turnover:
            turnovers = [
                r["portfolio_turnover_series"]
                for r in self.oos_results.values()
                if r.get("portfolio_turnover_series") is not None
            ]
            if turnovers:
                full_turnover = pd.concat(turnovers).sort_index()
                full_turnover = full_turnover[~full_turnover.index.duplicated(keep="first")]
                total_turnover = float(full_turnover.to_numpy(dtype=np.float64).sum())
                result["overall_avg_trade"] = (
                    float(vals.sum()) / total_turnover if total_turnover != 0 else np.nan
                )
        return result

    def run(self):
        """Convenience: in-sample, then out-of-sample, then aggregate."""
        self.run_in_sample()
        self.run_oos()
        return self.aggregate_oos()
