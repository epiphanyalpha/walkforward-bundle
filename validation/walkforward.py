# walkforward.py
import pandas as pd
from .selection_unit import SelectionUnit

class WalkForwardSchedule:
    def __init__(self, df, first_os, window_length, anchored=True, step_months=12):
        """
        Parameters:
          - df: DataFrame with DateTimeIndex.
          - first_os: The first out-of-sample date (string or pd.Timestamp). This is the boundary
                      between in-sample and OOS; the first in-sample slice is from (first_os - window_length)
                      to first_os.
          - window_length: In-sample window length (months).
          - anchored: If True, in-sample start is fixed at (first_os - window_length); if False, a fixed-length window rolls forward.
          - step_months: Step size (months) for moving the window.
        """
        self.df = df
        self.first_os = pd.to_datetime(first_os)
        self.window_length = window_length
        self.anchored = anchored
        self.step_months = step_months
        self.slices = self._generate_slices()

    def _generate_slices(self):
        # Compute the in-sample start based on first_os and window_length.
        analysis_start = self.first_os - pd.DateOffset(months=self.window_length)
        slices = []
        # Define maximum allowed end as the DataFrame's maximum date minus one step.
        max_allowed_end = self.df.index.max() - pd.DateOffset(months=self.step_months)
        if self.anchored:
            # Anchored: fixed start = analysis_start.
            slices.append((analysis_start, self.first_os))
            current_end = self.first_os + pd.DateOffset(months=self.step_months)
            while current_end <= max_allowed_end:
                slices.append((analysis_start, current_end))
                current_end += pd.DateOffset(months=self.step_months)
        else:
            # Unanchored (rolling): fixed window length.
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

class WalkForwardRunner:
    def __init__(self, df, schedule, risk_free_rate=0.0, metric_func=None,
                 top_n=10, max_corr=0.5, max_columns=10, turnover_df=None, min_avg_trade=None):
        """
        Parameters:
          - df: Returns DataFrame with DateTimeIndex.
          - schedule: A WalkForwardSchedule object.
          - risk_free_rate, metric_func, top_n, max_corr, max_columns: In-sample selection parameters.
          - turnover_df: Optional turnover DataFrame (aligned with df).
          - min_avg_trade: Optional minimum average trade threshold for in-sample selection.
        """
        self.df = df
        self.schedule = schedule
        self.risk_free_rate = risk_free_rate
        self.metric_func = metric_func
        self.top_n = top_n
        self.max_corr = max_corr
        self.max_columns = max_columns
        self.turnover_df = turnover_df
        self.min_avg_trade = min_avg_trade
        self.results = {}

    def run(self):
        """Run in-sample selection for each walk-forward slice."""
        for start, end in self.schedule.get_slices():
            df_slice = self.df.loc[start:end]
            turnover_slice = self.turnover_df.loc[start:end] if self.turnover_df is not None else None
            su = SelectionUnit(df_slice, self.risk_free_rate, turnover_df=turnover_slice, min_avg_trade=self.min_avg_trade)
            result = su.perform_selection(self.metric_func, self.top_n, self.max_corr, self.max_columns, metric_name="default")
            period_key = f"{start.date()} to {end.date()}"
            self.results[period_key] = result
        return self.results
