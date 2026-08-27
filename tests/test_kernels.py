"""
Kernel contracts: each compiled kernel must agree with an independent NumPy
reference, and must keep agreeing at the edges where hand-written loops break.
"""
import numpy as np
import pytest

from validation import kernels as K
from validation.metrics import METRICS, compute_calmar, compute_max_drawdown


@pytest.fixture(scope="module", autouse=True)
def _warm():
    K.warmup()


def _matrix(n=600, k=25, seed=0):
    return np.ascontiguousarray(np.random.default_rng(seed).normal(0, 0.012, (n, k)))


def _reference_drawdown(data, compounding=True):
    """Independent reference: equity starts at 1.0 and that value is a peak."""
    steps = 1 + data if compounding else data
    curve = np.cumprod(steps, axis=0) if compounding else np.cumsum(steps, axis=0)
    curve = np.vstack([np.full((1, data.shape[1]), 1.0 if compounding else 0.0), curve])
    peak = np.maximum.accumulate(curve, axis=0)
    dd = (peak - curve) / peak if compounding else (peak - curve)
    return dd.max(axis=0), np.sqrt((dd[1:] ** 2).mean(axis=0))


@pytest.mark.parametrize("compounding", [True, False])
def test_drawdown_matches_reference(compounding):
    d = _matrix()
    mdd, ulcer = K.drawdown_stats(d, compounding)
    ref_mdd, ref_ulcer = _reference_drawdown(d, compounding)
    np.testing.assert_allclose(mdd, ref_mdd, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(ulcer, ref_ulcer, rtol=1e-11, atol=1e-14)


def test_drawdown_counts_the_decline_from_inception():
    """The bug this kernel was written to fix.

    Equity goes 1.00 -> 0.50 -> 1.00. That is a 50% drawdown. Starting the
    running peak at the first bar's close instead of at 1.0 reports 0%.
    """
    d = np.array([[-0.5], [1.0]], dtype=np.float64)
    mdd, _ = K.drawdown_stats(d, True)
    assert mdd[0] == pytest.approx(0.5)
    assert compute_max_drawdown(d)[0] == pytest.approx(0.5)

    naive = np.cumprod(1 + d, axis=0)
    naive_dd = ((np.maximum.accumulate(naive, 0) - naive) / np.maximum.accumulate(naive, 0)).max()
    assert naive_dd == pytest.approx(0.0)      # what the old implementation said


def test_drawdown_monotone_series():
    up = np.linspace(0.001, 0.002, 400).reshape(-1, 1)
    assert K.drawdown_stats(up, True)[0][0] == pytest.approx(0.0, abs=1e-12)
    down = -up
    mdd = K.drawdown_stats(down, True)[0][0]
    assert 0.0 < mdd < 1.0


def test_standardize_gives_unit_norm_and_zero_mean():
    d = _matrix()
    z = K.standardize(d)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose((z * z).sum(axis=0), 1.0, rtol=1e-11)
    # dot product of standardized columns == Pearson correlation
    np.testing.assert_allclose(z[:, 0] @ z[:, 1], np.corrcoef(d[:, 0], d[:, 1])[0, 1],
                               rtol=1e-10)


def test_standardize_handles_constant_columns():
    d = np.ascontiguousarray(np.ones((50, 3)))
    z = K.standardize(d)
    assert np.all(z == 0.0)


def test_row_mean_subset_matches_numpy():
    d = _matrix()
    for cols in ([0], [0, 5, 9], list(range(25))):
        c = np.array(cols, dtype=np.int64)
        np.testing.assert_allclose(K.row_mean_subset(d, c), d[:, c].mean(axis=1),
                                   rtol=1e-12, atol=1e-15)


def test_row_mean_subset_empty():
    d = _matrix()
    out = K.row_mean_subset(d, np.array([], dtype=np.int64))
    assert out.shape == (d.shape[0],) and np.all(out == 0.0)


def test_greedy_respects_the_cap_it_was_given():
    d = _matrix(k=40, seed=3)
    z = K.standardize(d)
    for cap in (0.05, 0.3, 0.9):
        keep = K.greedy_uncorrelated(z, cap, 10, False)
        assert len(keep) <= 10 and keep[0] == 0
        corr = z[:, keep].T @ z[:, keep]
        off = corr[~np.eye(len(keep), dtype=bool)]
        assert off.size == 0 or off.max() < cap + 1e-12


def test_calmar_is_scale_free_in_both_terms():
    """Mean and drawdown are both per-period, so the ratio needs no annualisation."""
    d = _matrix(seed=7)
    got = compute_calmar(d)
    mdd = K.drawdown_stats(d, True)[0]
    want = np.where(mdd > 0, d.mean(axis=0) / np.where(mdd > 0, mdd, 1), 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-11)


def test_every_metric_honours_the_contract():
    """(T,k) in, (k,) out, plus an `ascending` attribute. No exceptions."""
    d = _matrix(k=12)
    for name, fn in METRICS.items():
        assert hasattr(fn, "ascending"), name
        if name in ("avg_trade", "information_ratio"):
            out = fn(d, np.abs(d)) if name == "avg_trade" else fn(d, d[:, 0])
        else:
            out = fn(d)
        out = np.asarray(out)
        assert out.shape == (d.shape[1],), f"{name} returned {out.shape}"
