"""
End-to-end example on a synthetic candidate universe.

The universe is built with a small number of latent factors plus noise, and a
mild spread in drift across candidates — enough structure that ranking and
de-correlation both have something to do, without a real edge to overfit to.

    python examples/quickstart.py
"""
import numpy as np
import pandas as pd

from validation import FullBacktesterEnsemble, generate_config_list


def synthetic_universe(n_rows=3000, n_cols=400, n_factors=8, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-01", periods=n_rows, freq="B")
    cols = [f"cand_{i:03d}" for i in range(n_cols)]

    factors = rng.normal(0, 0.009, size=(n_rows, n_factors))
    loadings = rng.normal(0, 1.0, size=(n_factors, n_cols))
    ret = factors @ loadings * 0.5 + rng.normal(0, 0.007, size=(n_rows, n_cols))
    ret += rng.normal(0.00008, 0.00012, size=n_cols)[None, :]   # dispersed drift

    turn = np.abs(rng.normal(0.04, 0.015, size=(n_rows, n_cols)))
    return (pd.DataFrame(ret, index=idx, columns=cols),
            pd.DataFrame(turn, index=idx, columns=cols))


def main():
    returns, turnover = synthetic_universe()
    print(f"universe: {returns.shape[0]} bars x {returns.shape[1]} candidates\n")

    grid = {
        "first_os": ["2016-01-01"],
        "window_length": [12, 24, 36, 48],
        "step_months": [3, 6, 12],
        "anchored": [True, False],
        "metric_name": ["sharpe", "calmar", "sortino"],
        "max_corr": [0.3, 0.5, 0.7],
        "max_columns": [5, 10],
        "top_n": [20],
        "risk_free_rate": [0.0],
        "min_avg_trade": [None],
        "include_turnover": [True],
    }
    configs = generate_config_list(grid)
    print(f"grid: {len(configs)} configurations\n")

    ensemble = FullBacktesterEnsemble(returns, turnover, configs)
    results, oos_series = ensemble.run(progress=False)

    print("--- the deliverable: the spread, not the best line ---")
    print(ensemble.summary().round(3).to_string())

    print("\n--- median annualized OOS Sharpe by in-sample window ---")
    print(results.groupby("window_length")["oos_sharpe_ann"].median().round(3).to_string())

    print("\n--- median annualized OOS Sharpe by ranking metric ---")
    print(results.groupby("metric_name")["oos_sharpe_ann"].median().round(3).to_string())

    best = results["oos_sharpe_ann"].idxmax()
    print(f"\nbest single configuration: {best}")
    print(f"  its Sharpe:      {results.loc[best, 'oos_sharpe_ann']:.2f}")
    print(f"  ensemble median: {results['oos_sharpe_ann'].median():.2f}")
    print("  <- the gap between these two numbers is what a single-curve")
    print("     back-test would have quietly kept to itself.")


if __name__ == "__main__":
    main()
