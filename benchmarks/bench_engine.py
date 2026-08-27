"""
Where the time actually goes.

Runs the same walk-forward grid three ways and reports wall-clock:

  legacy   pandas ``.loc`` slicing + pairwise correlation loop (v0.1 engine)
  bundle   pre-materialized matrices + standardize-once de-correlation
  bundle   + joblib fan-out across configurations

Run:  python benchmarks/bench_engine.py --rows 5000 --cols 1000
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from validation import FullBacktesterEnsemble, METRICS, generate_config_list
from validation._frames import FrameBundle

try:
    import numba as nb
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


# ------------------------------------------------------------------ legacy
if NUMBA_OK:
    @nb.njit(cache=True, fastmath=True)
    def _legacy_corr(col1, col2):
        n = len(col1)
        m1 = 0.0; m2 = 0.0
        for i in range(n):
            m1 += col1[i]; m2 += col2[i]
        m1 /= n; m2 /= n
        cov = 0.0; v1 = 0.0; v2 = 0.0
        for i in range(n):
            d1 = col1[i] - m1; d2 = col2[i] - m2
            cov += d1 * d2; v1 += d1 * d1; v2 += d2 * d2
        if v1 == 0.0 or v2 == 0.0:
            return 0.0
        return cov / np.sqrt(v1 * v2)

    @nb.njit(cache=True, fastmath=True)
    def _legacy_uncorrelated(data, max_corr, max_columns):
        k = data.shape[1]
        sel = [0]
        for c in range(1, k):
            ok = True
            for s in sel:
                if _legacy_corr(data[:, s], data[:, c]) >= max_corr:
                    ok = False
                    break
            if ok:
                sel.append(c)
            if len(sel) >= max_columns:
                break
        return np.array(sel)


def legacy_run(returns: pd.DataFrame, turnover: pd.DataFrame, configs, metric_name=None):
    """v0.1 engine: re-slice the DataFrame for every window of every config."""
    out = {}
    for cfg in configs:
        metric = METRICS[metric_name or cfg.get("metric_name", "sharpe")]
        first_os = pd.to_datetime(cfg["first_os"])
        wl, step = cfg["window_length"], cfg["step_months"]
        analysis_start = first_os - pd.DateOffset(months=wl)
        max_end = returns.index.max() - pd.DateOffset(months=step)

        ends, e = [first_os], first_os + pd.DateOffset(months=step)
        while e <= max_end:
            ends.append(e); e += pd.DateOffset(months=step)

        legs = []
        for end in ends:
            start = analysis_start if cfg["anchored"] else end - pd.DateOffset(months=wl)
            sl = returns.loc[start:end]                      # <- the hot line
            if sl.empty:
                continue
            data = sl.values
            vals = metric(data)
            top = np.argsort(vals)[::-1][:cfg["top_n"]]
            sub = np.ascontiguousarray(data[:, top].astype(np.float64))
            rel = _legacy_uncorrelated(sub, cfg["max_corr"], cfg["max_columns"])
            picked = returns.columns[top[rel]]

            oos = returns.loc[end + pd.Timedelta(days=1): end + pd.DateOffset(months=step)]
            if oos.empty or len(picked) == 0:
                continue
            legs.append(oos[picked].mean(axis=1))
        if legs:
            s = pd.concat(legs).sort_index()
            s = s[~s.index.duplicated(keep="first")]
            out[id(cfg)] = s
    return out


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--cols", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=args.rows, freq="D")
    cols = [f"Asset_{i}" for i in range(args.cols)]
    returns = pd.DataFrame(rng.uniform(-0.05, 0.05, (args.rows, args.cols)), idx, cols)
    turnover = pd.DataFrame(rng.uniform(0, 1, (args.rows, args.cols)), idx, cols)

    grid = {
        "first_os": ["2018-12-31"],
        "window_length": [12, 24, 36, 48],
        "step_months": [3, 6, 12],
        "anchored": [True, False],
        "risk_free_rate": [0.0],
        "top_n": [20],
        "max_corr": [0.3, 0.5, 0.7],
        "max_columns": [5, 10],
        "min_avg_trade": [None],
        "metric_name": ["sharpe", "momentum", "calmar", "sortino"],
        "include_turnover": [False],
    }
    configs = generate_config_list(grid)
    print(f"universe: {args.rows} x {args.cols}   configs: {len(configs)}")

    # warm the numba caches so we time the maths, not the JIT
    _ = legacy_run(returns.iloc[:400, :20], turnover.iloc[:400, :20],
                   [{**configs[0], "first_os": "2010-08-01", "window_length": 3,
                     "step_months": 3}])
    FullBacktesterEnsemble(returns.iloc[:400, :20], None,
                           [{**configs[0], "first_os": "2010-08-01",
                             "window_length": 3, "step_months": 3}]).run(progress=False)

    t0 = time.perf_counter(); legacy_run(returns, turnover, configs)
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    bundle = FrameBundle(returns, turnover)
    t_build = time.perf_counter() - t0

    ens = FullBacktesterEnsemble(bundle, None, configs)
    t0 = time.perf_counter(); ens.run(progress=False)
    t_new = time.perf_counter() - t0

    stats_cached = dict(bundle.metrics.stats())

    nocache = FrameBundle(returns, turnover, moments=False)
    ens_nc = FullBacktesterEnsemble(nocache, None, configs)
    t0 = time.perf_counter(); ens_nc.run(progress=False)
    t_nomoments = time.perf_counter() - t0

    bundle2 = FrameBundle(returns, turnover)
    ens2 = FullBacktesterEnsemble(bundle2, None, configs)
    t0 = time.perf_counter(); ens2.run(n_jobs=args.jobs, progress=False)
    t_par = time.perf_counter() - t0

    print(f"\n{'engine':<40}{'seconds':>10}{'speedup':>10}")
    print("-" * 60)
    print(f"{'legacy (.loc + pairwise corr)':<40}{t_legacy:>10.2f}{1.0:>10.1f}x")
    print(f"{'bundle, window cache only':<40}{t_nomoments:>10.2f}{t_legacy / t_nomoments:>9.1f}x")
    print(f"{'bundle, + prefix-sum moments':<40}{t_new:>10.2f}{t_legacy / t_new:>9.1f}x")
    print(f"{f'bundle, + joblib n_jobs={args.jobs}':<40}{t_par:>10.2f}{t_legacy / t_par:>9.1f}x")
    print(f"\nFrameBundle build: {t_build:.2f}s ({bundle.nbytes / 1e6:.0f} MB, paid once)")
    print(f"metric cache: {stats_cached['hits']} hits / {stats_cached['misses']} misses"
          f"  ({stats_cached['moment_cache_bytes'] / 1e6:.0f} MB prefix sums)")


if __name__ == "__main__":
    main()
