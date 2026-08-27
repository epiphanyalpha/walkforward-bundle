"""
The sixty-second demo.

    python -m validation.demo                  # synthetic prices, no downloads
    python -m validation.demo --csv spx.csv    # your own daily close series

Builds a candidate matrix from a parameter sweep, runs the full configuration
grid over it, and prints the one thing the library exists to show: how far
apart the out-of-sample results are, depending only on how you set the
walk-forward up.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .datasets import candidate_matrix, load_ohlcv, synthetic_prices
from .ensemble_backtester import FullBacktesterEnsemble, generate_config_list

GRID = {
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


def _rule(ch="-", n=64):
    return ch * n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m validation.demo",
                                 description=__doc__.strip().splitlines()[0])
    ap.add_argument("--csv", help="CSV with a date column and a close column")
    ap.add_argument("--first-os", default=None,
                    help="first out-of-sample date (default: 60%% through the data)")
    ap.add_argument("--plot", metavar="PATH", nargs="?", const="walkforward_bundle.png",
                    help="save the bundle chart to PATH (needs matplotlib)")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)

    if args.csv:
        prices = load_ohlcv(args.csv)
        source = f"{args.csv} ({len(prices)} bars)"
    else:
        prices = synthetic_prices()
        source = "synthetic prices — realistic texture, no edge by construction"

    print(_rule("="))
    print("  walk-forward bundle — demo")
    print(_rule("="))
    print(f"  data      {source}")
    print(f"  span      {prices.index[0].date()} to {prices.index[-1].date()}")

    returns, turnover = candidate_matrix(prices)
    per_cand = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    print(f"  candidates {returns.shape[1]} parameter sets "
          f"(annualized Sharpe {per_cand.min():.2f} to {per_cand.max():.2f}, "
          f"median {per_cand.median():.2f})")

    first_os = args.first_os or str(prices.index[int(len(prices) * 0.6)].date())
    configs = generate_config_list({**GRID, "first_os": [first_os]})
    print(f"  first OOS {first_os}")
    print(f"  grid      {len(configs)} walk-forward constructions\n")

    ens = FullBacktesterEnsemble(returns, turnover, configs, periods_per_year=252)
    results, oos = ens.run(n_jobs=args.jobs, progress=False)
    s = ens.summary()

    print(_rule())
    print("  ANNUALIZED OUT-OF-SAMPLE SHARPE, ACROSS THE GRID")
    print(_rule())
    for k in ["sharpe_min", "sharpe_q25", "sharpe_median", "sharpe_q75", "sharpe_max"]:
        bar = "#" * max(0, int(round((s[k] + 1) * 12)))
        print(f"  {k.replace('sharpe_', ''):>7}  {s[k]:+6.2f}  {bar}")
    print(f"\n  positive in {s['frac_positive']:.0%} of constructions"
          f"   |   spread (sd) {s['sharpe_std']:.2f}"
          f"   |   {s['total_run_time_sec']:.1f}s\n")

    best = results["oos_sharpe_ann"].idxmax()
    print(_rule())
    print(f"  best single construction   {results.loc[best, 'oos_sharpe_ann']:+.2f}"
          f"   {best}")
    print(f"  ensemble median            {s['sharpe_median']:+.2f}")
    print(_rule())
    best_v, med_v = results.loc[best, "oos_sharpe_ann"], s["sharpe_median"]
    if med_v <= 0 < best_v:
        n_pos = int((results["oos_sharpe_ann"] > 0).sum())
        print("  Every candidate here loses money after costs, and the median")
        print(f"  construction loses money too. {n_pos} construction(s) out of")
        print(f"  {len(results)} still print a positive out-of-sample Sharpe.")
        print("  That is the one that would have made it into the deck.\n")
    else:
        print("  The gap between those two numbers is what a single-curve")
        print("  back-test reports and does not mention.\n")

    print("  marginal effect of each construction choice (median Sharpe):")
    for axis in ["window_length", "step_months", "anchored", "metric_name"]:
        by = results.groupby(axis)["oos_sharpe_ann"].median()
        cells = "  ".join(f"{k}={v:+.2f}" for k, v in by.items())
        flag = "  <-- not a free choice" if (by.max() - by.min()) > 0.15 else ""
        print(f"    {axis:<14} {cells}{flag}")
    print()

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (--plot needs matplotlib: pip install matplotlib)", file=sys.stderr)
            return 0
        paths = ens.paths().fillna(0.0).cumsum(axis=1)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for row in paths.to_numpy():
            ax.plot(paths.columns, row, color="#2b3a55", alpha=0.07, lw=0.9)
        med_key = results["oos_sharpe_ann"].sort_values().index[len(results) // 2]
        ax.plot(paths.columns, paths.loc[med_key], color="#1d6a62", lw=2, label="median config")
        ax.plot(paths.columns, paths.loc[best], color="#a83c25", lw=2, label="best config")
        ax.axhline(0, color="#999", lw=.7, ls="--")
        ax.set_title(f"{len(configs)} walk-forward constructions, same strategy, same data")
        ax.set_ylabel("cumulative OOS return")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"  chart saved to {args.plot}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
