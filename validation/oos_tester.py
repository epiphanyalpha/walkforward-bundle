import numpy as np
import pandas as pd

def compute_cumulative_return(data):
    return np.prod(1 + data) - 1

def compute_oos_volatility(data, annualize=False, trading_days=252):
    vol = np.std(data)
    if annualize:
        vol *= np.sqrt(trading_days)
    return vol

def compute_oos_sharpe(data, risk_free_rate=0.0):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (mean_val - risk_free_rate) / std_val if std_val != 0 else np.nan

def compute_portfolio_avg_trade(portfolio_returns, portfolio_turnover):
    """
    Compute portfolio average trade as:
      average_trade = (sum of portfolio returns) / (sum of portfolio turnover)
    """
    total_pl = np.sum(portfolio_returns)
    total_turnover = np.sum(portfolio_turnover)
    return total_pl / total_turnover if total_turnover != 0 else np.nan

class OutOfSampleTester:
    def __init__(self, oos_df, selected_columns, turnover_oos_df=None, risk_free_rate=0.0):
        """
        Parameters:
          - oos_df: Out-of-sample returns DataFrame.
          - selected_columns: List of selected asset names.
          - turnover_oos_df: Out-of-sample turnover DataFrame.
          - risk_free_rate: Risk-free rate.
        """
        self.oos_df = oos_df
        self.selected_columns = selected_columns
        self.turnover_oos_df = turnover_oos_df
        self.risk_free_rate = risk_free_rate

    def run(self):
        data = self.oos_df[self.selected_columns]
        # Portfolio returns: equal-weight average.
        portfolio_returns = data.mean(axis=1)
        cum_return = compute_cumulative_return(portfolio_returns.values)
        vol = compute_oos_volatility(portfolio_returns.values, annualize=True)
        sharpe = compute_oos_sharpe(portfolio_returns.values, self.risk_free_rate)
        result = {
            "portfolio_returns_series": portfolio_returns,
            "cumulative_return": cum_return,
            "oos_volatility": vol,
            "oos_sharpe": sharpe
        }
        if self.turnover_oos_df is not None:
            portfolio_turnover = self.turnover_oos_df[self.selected_columns].mean(axis=1)
            result["portfolio_turnover_series"] = portfolio_turnover
            result["portfolio_avg_trade"] = compute_portfolio_avg_trade(portfolio_returns.values, portfolio_turnover.values)
        return result

if __name__ == "__main__":
    # Quick test on simulated OOS data.
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    np.random.seed(42)
    test_returns = np.random.randn(len(dates), 3).astype(np.float32) * 0.01
    df_oos = pd.DataFrame(test_returns, index=dates, columns=["Asset_A", "Asset_B", "Asset_C"])
    turnover_data = np.random.uniform(0, 0.02, size=test_returns.shape).astype(np.float32)
    df_turnover = pd.DataFrame(turnover_data, index=dates, columns=["Asset_A", "Asset_B", "Asset_C"])
    tester = OutOfSampleTester(df_oos, ["Asset_A", "Asset_B"], turnover_oos_df=df_turnover, risk_free_rate=0.0)
    performance = tester.run()
    print(performance)
