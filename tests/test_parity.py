"""
Parity and correctness tests.

The optimizations in this package are meant to be *invisible*: same numbers,
less time.  These tests pin that down by re-implementing the slow paths inline
and asserting equality.
"""
import numpy as np
import pandas as pd
import pytest

from validation import (
    FrameBundle, FullBacktester, FullBacktesterEnsemble, METRICS,
    generate_config_list, get_uncorrelated_indices,
)
from validation.initial_selector import _rank_top_n


# ---------------------------------------------------------------- fixtures
def make_universe(n_rows=1500, n_cols=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=n_rows, freq="D")
    cols = [f"cand_{i:03d}" for i in range(n_cols)]
    # a few genuinely correlated blocks, so de-correlation has something to do
    base = rng.normal(0, 0.01, size=(n_rows, 6))
    load = rng.normal(0, 1, size=(6, n_cols))
    ret = base @ load * 0.6 + rng.normal(0, 0.008, size=(n_rows, n_cols))
    ret += np.linspace(0, 0.0004, n_cols)[None, :]
    turn = np.abs(rng.normal(0.05, 0.02, size=(n_rows, n_cols)))
    return (pd.DataFrame(ret, index=idx, columns=cols),
            pd.DataFrame(turn, index=idx, columns=cols))


# ---------------------------------------------------------------- slicing
def test_bundle_slice_matches_pandas_loc():
    ret, turn = make_universe()
    b = FrameBundle(ret, turn, dtype=np.float64)
    for start, end in [("2014-03-01", "2015-06-30"),
                       ("2014-01-01", "2014-01-01"),
                       ("2013-01-01", "2014-02-01"),
                       ("2017-01-01", "2018-01-01")]:
        want = ret.loc[start:end]
        got, got_t = b.slice(start, end)
        assert got.values.shape == want.shape
        np.testing.assert_allclose(got.values, want.to_numpy(), rtol=0, atol=0)
        assert list(got.index) == list(want.index)
        np.testing.assert_allclose(got_t.values, turn.loc[start:end].to_numpy())


def test_bundle_slice_is_a_view_not_a_copy():
    """The whole point of FrameBundle: slicing must not allocate.

    Assert on shared memory, not on ``.base`` identity — depending on the
    pandas version, ``to_numpy`` may hand back a view of an internal block, so
    the base chain has an extra link even though nothing was copied.
    """
    ret, _ = make_universe(n_rows=300, n_cols=10)
    b = FrameBundle(ret, dtype=np.float64)
    view, _ = b.slice("2014-02-01", "2014-05-01")
    assert np.shares_memory(view.values, b.returns)
    assert view.values.shape[0] < b.returns.shape[0]


# ------------------------------------------------------- correlation filter
def _reference_uncorrelated(data, max_corr, max_columns):
    """The original O(k^2 T) loop, recomputing moments on every pair."""
    def corr(a, b):
        m1, m2 = a.mean(), b.mean()
        d1, d2 = a - m1, b - m2
        v1, v2 = (d1 * d1).sum(), (d2 * d2).sum()
        if v1 == 0 or v2 == 0:
            return 0.0
        return float((d1 * d2).sum() / np.sqrt(v1 * v2))

    keep = [0]
    for c in range(1, data.shape[1]):
        if all(corr(data[:, s], data[:, c]) < max_corr for s in keep):
            keep.append(c)
        if len(keep) >= max_columns:
            break
    return np.array(keep)


@pytest.mark.parametrize("max_corr", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("max_columns", [3, 10])
def test_decorrelation_matches_reference(max_corr, max_columns):
    ret, _ = make_universe(n_rows=800, n_cols=40, seed=3)
    data = ret.to_numpy(dtype=np.float64)
    want = _reference_uncorrelated(data, max_corr, max_columns)
    got = get_uncorrelated_indices(data, max_corr, max_columns)
    np.testing.assert_array_equal(got, want)


def test_decorrelation_handles_degenerate_input():
    assert get_uncorrelated_indices(np.zeros((0, 0)), 0.5, 5).size == 0
    const = np.ones((50, 4))
    # constant columns correlate 0.0 with everything -> all kept, up to the cap
    assert get_uncorrelated_indices(const, 0.5, 3).size == 3


# --------------------------------------------------------------- ranking
def test_ranking_does_not_promote_nans():
    # NaN *and* +/-inf are demoted: an infinite Sharpe means a zero-variance
    # candidate, which is a data artefact rather than the best strategy.
    v = np.array([0.4, np.nan, 0.9, np.inf, 0.7])
    top = _rank_top_n(v, 3, ascending=False)
    np.testing.assert_array_equal(top, np.array([2, 4, 0]))
    v2 = np.array([0.4, np.nan, 0.9, 0.7])
    np.testing.assert_array_equal(_rank_top_n(v2, 2, ascending=False), np.array([2, 3]))
    # ascending: NaN must not win the "lowest" slot either
    np.testing.assert_array_equal(_rank_top_n(v2, 2, ascending=True), np.array([0, 3]))


# --------------------------------------------------------------- end to end
def test_walk_forward_is_causal():
    """No in-sample window may overlap the period it is judged on."""
    ret, turn = make_universe()
    fb = FullBacktester(ret, turn, first_os="2016-01-01", window_length=24,
                        step_months=6, metric_func=METRICS["sharpe"])
    fb.run_in_sample()
    fb.run_oos()
    assert fb.oos_results
    for (start, is_end) in fb.oos_results:
        oos = fb.oos_results[(start, is_end)]["portfolio_returns_series"]
        assert oos.index.min() > is_end


def test_ensemble_runs_and_reports_a_spread():
    ret, turn = make_universe()
    grid = {
        "first_os": ["2016-01-01"],
        "window_length": [12, 24],
        "step_months": [6, 12],
        "anchored": [True, False],
        "risk_free_rate": [0.0],
        "top_n": [10],
        "max_corr": [0.5],
        "max_columns": [5],
        "min_avg_trade": [None],
        "metric_name": ["sharpe", "momentum"],
        "include_turnover": [True],
    }
    configs = generate_config_list(grid)
    assert len(configs) == 16

    ens = FullBacktesterEnsemble(ret, turn, configs)
    results, series = ens.run(progress=False)
    assert len(results) == 16
    assert results["oos_sharpe"].notna().all()
    assert all(isinstance(s, pd.Series) and len(s) for s in series.values())

    summary = ens.summary()
    assert summary["n_configs"] == 16
    assert summary["sharpe_min"] <= summary["sharpe_median"] <= summary["sharpe_max"]

    paths = ens.paths()
    assert paths.shape[0] == 16


def test_parallel_matches_serial():
    ret, turn = make_universe(n_rows=900, n_cols=30, seed=7)
    grid = {
        "first_os": ["2015-06-01"],
        "window_length": [12, 18],
        "step_months": [6],
        "anchored": [True],
        "risk_free_rate": [0.0],
        "top_n": [8],
        "max_corr": [0.6],
        "max_columns": [4],
        "min_avg_trade": [None],
        "metric_name": ["sharpe", "calmar"],
        "include_turnover": [False],
    }
    configs = generate_config_list(grid)
    serial = FullBacktesterEnsemble(ret, turn, configs).run(progress=False)[0]
    par = FullBacktesterEnsemble(ret, turn, configs).run(n_jobs=2, progress=False)[0]
    pd.testing.assert_series_equal(
        serial["oos_sharpe"].sort_index(), par["oos_sharpe"].sort_index()
    )


def test_unknown_metric_is_rejected():
    ret, turn = make_universe(n_rows=400, n_cols=10)
    cfg = [{"first_os": "2014-09-01", "window_length": 6, "step_months": 3,
            "metric_name": "not_a_metric"}]
    with pytest.raises(ValueError, match="Unknown metric"):
        FullBacktesterEnsemble(ret, turn, cfg).run(progress=False)


# --------------------------------------------------------- moment fast path
def test_moment_metrics_match_direct_computation():
    """Prefix sums must reproduce the textbook formulas, not approximate them."""
    from validation.moments import MetricEngine

    ret, _ = make_universe(n_rows=2000, n_cols=80, seed=11)
    bundle = FrameBundle(ret, dtype=np.float64)
    eng = MetricEngine(bundle, moments=True)
    raw = MetricEngine(bundle, moments=False)

    for name in ["sharpe", "volatility", "highest_return", "momentum"]:
        fn = METRICS[name]
        for i0, i1 in [(0, 2000), (100, 400), (0, 13), (1500, 2000)]:
            fast = eng.compute(name, fn, i0, i1, risk_free_rate=0.001, lookback=12)
            slow = raw.compute(name, fn, i0, i1, risk_free_rate=0.001, lookback=12)
            # rtol is set by the *reference*, not the fast path: several of
            # the shipped metrics return float32, which is ~7 digits.
            np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-12,
                                       err_msg=f"{name} [{i0}:{i1}]")


def test_moment_sharpe_is_precise_in_float64():
    """Against a full-precision reference, the prefix-sum path holds ~1e-12."""
    from validation.moments import MetricEngine

    ret, _ = make_universe(n_rows=4000, n_cols=50, seed=13)
    bundle = FrameBundle(ret, dtype=np.float64)
    eng = MetricEngine(bundle, moments=True)
    for i0, i1 in [(0, 4000), (250, 3900), (3000, 4000)]:
        x = ret.to_numpy(dtype=np.float64)[i0:i1]
        want = (x.mean(0) - 0.0005) / x.std(0)
        got = eng.compute("sharpe", METRICS["sharpe"], i0, i1, risk_free_rate=0.0005)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_metric_cache_does_not_change_results():
    ret, turn = make_universe(n_rows=1200, n_cols=40, seed=5)
    grid = {
        "first_os": ["2015-06-01"],
        "window_length": [12, 18],
        "step_months": [6],
        "anchored": [True],
        "risk_free_rate": [0.0],
        "top_n": [10],
        "max_corr": [0.4, 0.7],
        "max_columns": [5],
        "min_avg_trade": [None],
        "metric_name": ["sharpe", "max_drawdown"],
        "include_turnover": [True],
    }
    configs = generate_config_list(grid)
    cached = FullBacktesterEnsemble(ret, turn, configs).run(progress=False)[0]

    plain_bundle = FrameBundle(ret, turn, moments=False)
    plain = FullBacktesterEnsemble(plain_bundle, None, configs).run(progress=False)[0]

    pd.testing.assert_frame_equal(
        cached[["oos_sharpe", "oos_return_compounded"]].sort_index(),
        plain[["oos_sharpe", "oos_return_compounded"]].sort_index(),
    )


def test_cache_actually_hits_across_the_grid():
    """max_corr / max_columns must not force a metric recomputation."""
    ret, _ = make_universe(n_rows=1200, n_cols=40, seed=5)
    bundle = FrameBundle(ret)
    grid = {
        "first_os": ["2015-06-01"],
        "window_length": [12],
        "step_months": [6],
        "anchored": [True],
        "risk_free_rate": [0.0],
        "top_n": [10],
        "max_corr": [0.3, 0.5, 0.7],
        "max_columns": [3, 5],
        "min_avg_trade": [None],
        "metric_name": ["sharpe"],
        "include_turnover": [False],
    }
    FullBacktesterEnsemble(bundle, None, generate_config_list(grid)).run(progress=False)
    stats = bundle.metrics.stats()
    # 6 configs share one schedule: one evaluation per window, five cache hits.
    assert stats["hits"] == 5 * stats["misses"]
