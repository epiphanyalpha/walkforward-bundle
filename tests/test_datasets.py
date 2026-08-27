"""
The demo path: candidate construction must be causal, and the format it
produces must be the format the ensemble consumes.
"""
import numpy as np
import pandas as pd

from validation import FullBacktesterEnsemble, generate_config_list
from validation.datasets import candidate_matrix, load_ohlcv, synthetic_prices


def test_synthetic_prices_are_deterministic_and_sane():
    a = synthetic_prices(n=500, seed=3)
    b = synthetic_prices(n=500, seed=3)
    pd.testing.assert_series_equal(a, b)
    assert (a > 0).all()
    assert a.index.is_monotonic_increasing
    ret = a.pct_change().dropna()
    # vol clustering: |r| must be autocorrelated even though r is not
    assert ret.abs().autocorr(1) > ret.autocorr(1)


def test_candidate_matrix_shapes_and_alignment():
    px = synthetic_prices(n=1500, seed=1)
    r, t = candidate_matrix(px)
    assert r.shape == t.shape
    assert list(r.columns) == list(t.columns)
    assert r.index.equals(px.index) and t.index.equals(px.index)
    assert np.isfinite(r.to_numpy()).all()
    assert (t.to_numpy() >= 0).all()


def test_candidate_returns_are_causal():
    """Perturbing the last bar must not change any earlier candidate return.

    This is the property that matters: if it fails, every downstream
    walk-forward number is contaminated no matter how careful the schedule is.
    """
    px = synthetic_prices(n=800, seed=4)
    base, _ = candidate_matrix(px)

    bumped = px.copy()
    bumped.iloc[-1] *= 1.25
    after, _ = candidate_matrix(bumped)

    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_costs_actually_bite():
    px = synthetic_prices(n=1500, seed=2)
    cheap, _ = candidate_matrix(px, cost_bps=0.0)
    dear, _ = candidate_matrix(px, cost_bps=20.0)
    assert (dear.sum() < cheap.sum()).all()


def test_matrix_feeds_the_ensemble_unmodified():
    px = synthetic_prices(n=2200, seed=6)
    r, t = candidate_matrix(px, fasts=(5, 20), slows=(60, 160))
    configs = generate_config_list({
        "first_os": [str(px.index[1200].date())],
        "window_length": [12, 24], "step_months": [6], "anchored": [True],
        "metric_name": ["sharpe"], "max_corr": [0.5], "max_columns": [3],
        "top_n": [6], "risk_free_rate": [0.0], "min_avg_trade": [None],
        "include_turnover": [True],
    })
    results, series = FullBacktesterEnsemble(r, t, configs, periods_per_year=252).run(progress=False)
    assert len(results) == 2
    assert results["oos_sharpe_ann"].notna().all()
    assert results["oos_avg_trade"].notna().all()


def test_load_ohlcv_sniffs_columns(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("Data,Open,Close\n2020-01-01,1,10\n2020-01-02,1,11\nbad,,\n")
    s = load_ohlcv(p)
    assert list(s.values) == [10.0, 11.0]
    assert s.name == "CLOSE"


def test_demo_runs_end_to_end(capsys):
    from validation.demo import main
    rc = main(["--first-os", "2005-01-01"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ANNUALIZED OUT-OF-SAMPLE SHARPE" in out
    assert "ensemble median" in out
