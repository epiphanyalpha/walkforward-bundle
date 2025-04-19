# metrics.py
import numpy as np

def compute_sharpe(data, risk_free_rate=0.0):
    """Compute the Sharpe ratio for each asset."""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    sharpe = np.zeros(data.shape[1], dtype=np.float32)
    mask = stds > 0
    sharpe[mask] = (means[mask] - risk_free_rate) / stds[mask]
    return sharpe
compute_sharpe.ascending = False  # Higher Sharpe is better

def compute_highest_return(data):
    """Compute the total return (sum of returns) for each asset."""
    return np.sum(data, axis=0)
compute_highest_return.ascending = False  # Higher return is better

def compute_max_drawdown(data):
    """
    Compute maximum drawdown for each asset.
    Returns positive drawdown values.
    """
    cum_returns = np.cumprod(1 + data, axis=0)
    running_max = np.maximum.accumulate(cum_returns, axis=0)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = np.min(drawdowns, axis=0)
    return -max_dd
compute_max_drawdown.ascending = True  # Lower drawdown is better

def compute_volatility(data, annualize=False, trading_days=252):
    """Compute volatility (standard deviation) for each asset."""
    vol = np.std(data, axis=0)
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol
compute_volatility.ascending = False  # Lower volatility is better

def compute_momentum(data, lookback=12):
    """
    Compute momentum as cumulative return over the last `lookback` periods.
    """
    if data.shape[0] < lookback:
        lookback = data.shape[0]
    cum_return = np.prod(1 + data[-lookback:], axis=0) - 1
    return cum_return
compute_momentum.ascending = False  # Higher momentum is better

def compute_average_trade_ratio(returns, turnover, risk_free_rate=0.0):
    """
    Compute the average trade for each asset as:
       average_trade = (sum of P&L) / (sum of turnover)
    Parameters:
      - returns: numpy array of shape (time, assets) representing P&L.
      - turnover: numpy array of shape (time, assets) representing daily turnover.
    """
    total_pl = np.sum(returns, axis=0)
    total_turnover = np.sum(turnover, axis=0)
    avg_trade = np.where(total_turnover != 0, total_pl / total_turnover, np.nan)
    return avg_trade
compute_average_trade_ratio.ascending = False  # Higher average trade ratio is better

def compute_composite(data, risk_free_rate=0.0, momentum_lookback=12, weight_sharpe=0.7, weight_momentum=0.3):
    """Compute a composite metric as a weighted combination of Sharpe and momentum."""
    sharpe = compute_sharpe(data, risk_free_rate)
    momentum = compute_momentum(data, lookback=momentum_lookback)
    return weight_sharpe * sharpe + weight_momentum * momentum
compute_composite.ascending = False  # Higher composite is better

# Expose a dictionary mapping names to metric functions.
METRICS = {
    "sharpe": compute_sharpe,
    "highest_return": compute_highest_return,
    "max_drawdown": compute_max_drawdown,
    "volatility": compute_volatility,
    "momentum": compute_momentum,
    "avg_trade": compute_average_trade_ratio,
    "composite": compute_composite
}

if __name__ == "__main__":
    # Quick test on dummy data.
    np.random.seed(0)
    data = np.random.randn(252, 5).astype(np.float32) * 0.01
    print("Sharpe:", compute_sharpe(data))
    print("Avg Trade Ratio (dummy turnover):", compute_average_trade_ratio(data, np.abs(data)))
