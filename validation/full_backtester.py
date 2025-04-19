# full_backtester.py
import pandas as pd
import numpy as np
from .walkforward import WalkForwardSchedule, WalkForwardRunner
from .oos_tester import OutOfSampleTester

class FullBacktester:
    def __init__(self, df, turnover_df,
                 first_os, window_length, step_months=12, anchored=True,
                 risk_free_rate=0.0, metric_func=None, top_n=10,
                 max_corr=0.5, max_columns=10, min_avg_trade=None):
        """
        Parameters:
          - df: Returns DataFrame covering the entire period.
          - turnover_df: Turnover DataFrame (aligned with df).
          - first_os: The first out-of-sample boundary (string or pd.Timestamp).  
                     The first in-sample slice is from (first_os - window_length) to first_os.
          - window_length: In-sample window length in months.
          - step_months: Step size (months); also defines the OOS period length.
          - anchored: If True, in-sample start is fixed; if False, a fixed window rolls forward.
          - risk_free_rate, metric_func, top_n, max_corr, max_columns: In-sample selection parameters.
          - min_avg_trade: Optional minimum average trade threshold (in-sample).
        """
        self.df = df
        self.turnover_df = turnover_df
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
        # Build in-sample schedule.
        #from .walkforward import WalkForwardSchedule
        self.schedule = WalkForwardSchedule(df, first_os, window_length, anchored=anchored, step_months=step_months)
        self.in_sample_results = {}
        self.oos_results = {}

    def run_in_sample(self):
        """Run in-sample selection on each walk-forward slice."""
        #from walkforward import WalkForwardRunner
        runner = WalkForwardRunner(self.df, self.schedule, risk_free_rate=self.risk_free_rate,
                                   metric_func=self.metric_func, top_n=self.top_n,
                                   max_corr=self.max_corr, max_columns=self.max_columns,
                                   turnover_df=self.turnover_df, min_avg_trade=self.min_avg_trade)
        self.in_sample_results = runner.run()
        return self.in_sample_results

    def run_oos(self):
        """
        For each in-sample slice, define the out-of-sample period as:
          from (in-sample end + 1 day) to (in-sample end + step_months)
        Run the OOS tester on that slice using the selected assets.
        """
        for period, sel in self.in_sample_results.items():
            # period is of the form "YYYY-MM-DD to YYYY-MM-DD"
            insample_end = pd.to_datetime(period.split(" to ")[1])
            oos_start = insample_end + pd.Timedelta(days=1)
            oos_end = insample_end + pd.DateOffset(months=self.step_months)
            oos_slice = self.df.loc[oos_start:oos_end]
            oos_turnover = self.turnover_df.loc[oos_start:oos_end]
            if oos_slice.empty:
                continue
            selected_assets = sel["filtered"]
            if len(selected_assets) == 0:
                continue
            tester = OutOfSampleTester(oos_slice, selected_assets, turnover_oos_df=oos_turnover,
                                       risk_free_rate=self.risk_free_rate)
            self.oos_results[period] = tester.run()
        return self.oos_results

    def aggregate_oos(self):
        """
        Concatenate the portfolio return series and the portfolio turnover series from each OOS period,
        then compute overall portfolio metrics, including overall average trade:
        
            overall_avg_trade = sum(full portfolio returns) / sum(full portfolio turnover)
        """
        portfolio_returns_list = []
        portfolio_turnover_list = []
        for period, res in self.oos_results.items():
            pr_series = res.get("portfolio_returns_series")
            pt_series = res.get("portfolio_turnover_series")
            if pr_series is not None and not pr_series.empty:
                portfolio_returns_list.append(pr_series)
            if pt_series is not None and not pt_series.empty:
                portfolio_turnover_list.append(pt_series)
        if not portfolio_returns_list:
            return None
        full_returns_series = pd.concat(portfolio_returns_list).sort_index()
        overall_cum_return = np.prod(1 + full_returns_series.values) - 1
        overall_vol = np.std(full_returns_series.values)
        overall_sharpe = (np.mean(full_returns_series.values) - self.risk_free_rate) / overall_vol if overall_vol != 0 else np.nan

        overall_avg_trade = None
        if portfolio_turnover_list:
            full_turnover_series = pd.concat(portfolio_turnover_list).sort_index()
            overall_avg_trade = np.sum(full_returns_series.values) / np.sum(full_turnover_series.values) if np.sum(full_turnover_series.values) != 0 else np.nan

        return {
            "full_oos_series": full_returns_series,
            "overall_cumulative_return": overall_cum_return,
            "overall_volatility": overall_vol,
            "overall_sharpe": overall_sharpe,
            "overall_avg_trade": overall_avg_trade
        }


if __name__ == "__main__":
    # Test the full backtester with simulated data.
    date_range = pd.date_range(start="2015-01-01", end="2025-12-31", freq="B")
    n_cols = 90
    np.random.seed(42)
    returns_data = np.random.randn(len(date_range), n_cols).astype(np.float32)
    df = pd.DataFrame(returns_data, index=date_range, columns=[f"col_{i}" for i in range(n_cols)])
    turnover_data = np.random.uniform(0, 0.02, returns_data.shape).astype(np.float32)
    turnover_df = pd.DataFrame(turnover_data, index=date_range, columns=df.columns)
    
    first_os = "2016-12-31"
    window_length = 12
    step_months = 12
    anchored = True
    risk_free_rate = 0.0
    top_n = 10
    max_corr = 0.5
    max_columns = 10
    min_avg_trade = 0.5  # example threshold
    
    from metrics import METRICS
    metric_name = "composite"
    metric_func = METRICS[metric_name]
    
    backtester = FullBacktester(df, turnover_df,
                                  first_os, window_length, step_months, anchored,
                                  risk_free_rate, metric_func, top_n, max_corr, max_columns, min_avg_trade)
    print("Running in-sample selection...")
    backtester.run_in_sample()
    print("Running out-of-sample testing...")
    backtester.run_oos()
    agg = backtester.aggregate_oos()
    print("Aggregated OOS performance:")
    print(agg)
