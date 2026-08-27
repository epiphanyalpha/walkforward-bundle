"""
Units. The configuration speaks calendar months; the kernels speak bars. Every
place those two meet is a place a number can silently mean the wrong thing.
"""
import numpy as np
import pandas as pd
import pytest

from validation import FrameBundle
from validation.metrics import compute_momentum


@pytest.mark.parametrize("freq,expected", [("D", 365), ("B", 261), ("h", 8766), ("W", 52)])
def test_bars_per_year_is_measured_not_assumed(freq, expected):
    idx = pd.date_range("2015-01-01", periods=3000, freq=freq)
    b = FrameBundle(pd.DataFrame(np.zeros((3000, 2)), idx, ["a", "b"]))
    assert b.bars_per_year == pytest.approx(expected, rel=0.02)


def test_business_days_are_not_mistaken_for_calendar_days():
    """The median gap between business days is one day; the rate is not 365/yr.

    Measuring the span rather than the gap is what keeps a 12-month lookback
    from becoming a 17-month one on a business-day index.
    """
    idx = pd.date_range("2015-01-01", periods=2000, freq="B")
    b = FrameBundle(pd.DataFrame(np.zeros((2000, 2)), idx, ["a", "b"]))
    assert b.bars_per_year < 300
    assert b.bars_for_months(12) == pytest.approx(261, rel=0.02)


def test_months_convert_consistently():
    idx = pd.date_range("2015-01-01", periods=4000, freq="D")
    b = FrameBundle(pd.DataFrame(np.zeros((4000, 2)), idx, ["a", "b"]))
    assert b.bars_for_months(12) == pytest.approx(365, rel=0.01)
    assert b.bars_for_months(24) == pytest.approx(730, rel=0.01)
    assert b.bars_for_months(0.25) >= 1          # never degenerates to zero


def test_momentum_reads_bars_not_months():
    """A 24 in `lookback_bars` must mean 24 bars, and be visible as such."""
    data = np.zeros((100, 1))
    data[-10:, 0] = 1.0
    assert compute_momentum(data, lookback_bars=10)[0] == pytest.approx(10.0)
    assert compute_momentum(data, lookback_bars=5)[0] == pytest.approx(5.0)
    assert compute_momentum(data, lookback_bars=500)[0] == pytest.approx(10.0)  # clamped


def test_momentum_months_reaches_the_metric_as_bars():
    """End-to-end: the config says months, the kernel receives bars."""
    from validation import FullBacktesterEnsemble, generate_config_list

    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-01-01", periods=1800, freq="D")
    cols = [f"c{i}" for i in range(12)]
    r = pd.DataFrame(rng.normal(0, 0.01, (1800, 12)), idx, cols)

    base = dict(first_os=["2016-01-01"], window_length=[24], step_months=[6],
                anchored=[True], metric_name=["momentum"], max_corr=[0.6],
                max_columns=[4], top_n=[8], risk_free_rate=[0.0],
                min_avg_trade=[None], include_turnover=[False])

    bundle = FrameBundle(r)
    ens = FullBacktesterEnsemble(bundle, None,
                                 generate_config_list({**base, "momentum_months": [12]}))
    ens.run(progress=False)
    keys = [k for k in bundle.metrics._cache]
    assert keys, "the metric cache should have been populated"
    lookbacks = {k[-1] for k in keys}
    # 12 months on a daily index is ~365 bars, emphatically not 12
    assert lookbacks == {bundle.bars_for_months(12)}
    assert max(lookbacks) > 300
