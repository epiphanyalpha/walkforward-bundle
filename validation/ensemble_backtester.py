"""
The ensemble: run the whole configuration grid, keep every OOS path.

The point of the library.  A single walk-forward gives you one number and no
way to tell whether it is the strategy talking or the choice of window length.
Running the grid turns that one number into a distribution you can actually
reason about.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._frames import FrameBundle
from .full_backtester import FullBacktester
from .metrics import METRICS

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(it, **kwargs):
        return it

__all__ = ["FullBacktesterEnsemble", "generate_config_list", "config_key"]


DEFAULTS = {
    "first_os": None,
    "window_length": 24,
    "step_months": 12,
    "anchored": True,
    "risk_free_rate": 0.0,
    "top_n": 10,
    "max_corr": 0.5,
    "max_columns": 10,
    "min_avg_trade": None,
    "metric_name": "sharpe",
    "include_turnover": False,
    "momentum_months": 12,
}


def config_key(config: dict) -> str:
    """Stable, human-readable identifier for one grid point."""
    return (
        f"{config['metric_name']}_WL{config['window_length']}"
        f"_Step{config['step_months']}_Corr{config['max_corr']}"
        f"_Cols{config['max_columns']}_Turn{config.get('include_turnover', False)}"
        f"_Anch{config.get('anchored', True)}"
    )


def generate_config_list(config_grid: Dict[str, Iterable]) -> List[dict]:
    """Cartesian product of the grid, as a list of config dicts.

    Returns a list rather than a generator so the caller can check the size
    before committing to a sweep, and so the grid can be reused.
    """
    keys = list(config_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(config_grid[k] for k in keys))]


def _selection_frame(in_sample_results: dict) -> pd.DataFrame:
    """Per-fold record of what was actually chosen, tidy and inspectable.

    One row per (window, admitted candidate), in admission order — which is
    also rank order, since the de-correlation walks the shortlist best-first.
    """
    rows = []
    for (start, end), sel in in_sample_results.items():
        chosen = sel.get("filtered") or []
        values = sel.get("filtered_values") or []
        avg_trade = sel.get("avg_trade")
        for i, name in enumerate(chosen):
            row = {
                "window_start": start,
                "window_end": end,
                "rank": i + 1,
                "candidate": name,
                "metric_value": values[i] if i < len(values) else np.nan,
                "n_admitted": len(chosen),
            }
            if avg_trade is not None and i < len(avg_trade):
                row["is_avg_trade"] = avg_trade[i]
            rows.append(row)
    return pd.DataFrame(rows)


def _run_one(bundle: FrameBundle, config: dict, periods_per_year: int,
             keep_selections: bool = False):
    """Execute a single configuration. Kept module-level so it can be pickled."""
    cfg = {**DEFAULTS, **config}
    key = config_key(cfg)
    t0 = time.perf_counter()

    metric_func = METRICS.get(cfg["metric_name"])
    if metric_func is None:
        raise ValueError(
            f"Unknown metric {cfg['metric_name']!r}. Available: {sorted(METRICS)}"
        )

    fb = FullBacktester(
        bundle, None,
        first_os=cfg["first_os"],
        window_length=cfg["window_length"],
        step_months=cfg["step_months"],
        anchored=cfg["anchored"],
        risk_free_rate=cfg["risk_free_rate"],
        metric_func=metric_func,
        top_n=cfg["top_n"],
        max_corr=cfg["max_corr"],
        max_columns=cfg["max_columns"],
        min_avg_trade=cfg["min_avg_trade"],
        include_turnover=cfg["include_turnover"],
        periods_per_year=periods_per_year,
        metric_name=cfg["metric_name"],
        momentum_months=cfg["momentum_months"],
    )
    agg = fb.run() or {}
    selections = _selection_frame(fb.in_sample_results) if keep_selections else None

    record = dict(cfg)
    record["config_key"] = key
    sharpe = agg.get("overall_sharpe")
    record.update({
        "oos_return_compounded": agg.get("overall_return_compounded"),
        "oos_return_additive": agg.get("overall_return_additive"),
        "oos_volatility": agg.get("overall_volatility"),
        # per-period, in the units of the input bars ...
        "oos_sharpe": sharpe,
        # ... and the annualized version, so nobody has to guess which is which
        "oos_sharpe_ann": sharpe * np.sqrt(periods_per_year)
        if sharpe is not None and np.isfinite(sharpe) else None,
        "oos_avg_trade": agg.get("overall_avg_trade"),
        "n_oos_periods": agg.get("n_periods", 0),
        "run_time_sec": round(time.perf_counter() - t0, 4),
    })
    return key, record, agg.get("full_oos_series"), selections


def _schedule_key(cfg: dict) -> tuple:
    """Configs sharing this key share their in-sample windows — and therefore
    every cached metric evaluation."""
    c = {**DEFAULTS, **cfg}
    return (str(c["first_os"]), c["window_length"], c["step_months"],
            c["anchored"], c["metric_name"], c["risk_free_rate"])


def _chunk_by_schedule(configs: List[dict], n_chunks: int) -> List[List[dict]]:
    """Group configs by schedule, then pack the groups into ``n_chunks`` bins.

    Dispatching one config per task looks like the obvious parallelisation and
    is the wrong one: every task ships a fresh copy of the universe and starts
    with an empty metric cache, so the work that the cache was eliminating
    comes straight back — multiplied by the number of tasks. Keeping
    schedule-siblings together in one task preserves the cache inside the
    worker.
    """
    groups: Dict[tuple, List[dict]] = {}
    for cfg in configs:
        groups.setdefault(_schedule_key(cfg), []).append(cfg)

    ordered = sorted(groups.values(), key=len, reverse=True)
    n_chunks = max(1, min(n_chunks, len(ordered)))
    bins: List[List[dict]] = [[] for _ in range(n_chunks)]
    sizes = [0] * n_chunks
    for grp in ordered:  # greedy longest-processing-time-first packing
        i = sizes.index(min(sizes))
        bins[i].extend(grp)
        sizes[i] += len(grp)
    return [b for b in bins if b]


def _run_chunk(bundle: FrameBundle, configs: List[dict], periods_per_year: int,
               keep_selections: bool = False):
    """Run a group of configurations in one worker, sharing one metric cache."""
    return [_run_one(bundle, cfg, periods_per_year, keep_selections) for cfg in configs]


class FullBacktesterEnsemble:
    """Run a grid of walk-forward configurations over one candidate universe.

    Parameters
    ----------
    df : pd.DataFrame
        Returns, rows = timestamps, columns = candidate strategies or assets.
    turnover_df : pd.DataFrame, optional
        Turnover aligned with ``df``. Required by the ``avg_trade`` metric and
        by any config with ``include_turnover=True`` or ``min_avg_trade``.
    config_list : iterable of dict
        Usually the output of :func:`generate_config_list`.
    periods_per_year : int
        Only used for annualizing the reported volatility (252 for daily bars,
        365 for crypto, 24*365 for hourly).
    dtype : np.dtype
        Storage dtype for the pre-materialized matrices.

    Attributes set by :meth:`run`
    -----------------------------
    results_df : pd.DataFrame
        One row per configuration, indexed by ``config_key``.
    oos_series : dict
        ``config_key -> pd.Series`` of the stitched out-of-sample returns.
    selections : dict
        ``config_key -> pd.DataFrame`` of what was chosen in every fold, when
        ``run(keep_selections=True)``.
    """

    def __init__(self, df, turnover_df=None, config_list=(), periods_per_year: int = 252,
                 dtype=np.float32):
        self.bundle = df if isinstance(df, FrameBundle) else FrameBundle(df, turnover_df, dtype=dtype)
        self.config_list = list(config_list)
        self.periods_per_year = periods_per_year
        self.oos_series: Dict[str, pd.Series] = {}
        self.selections: Dict[str, pd.DataFrame] = {}
        self.results_df: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.config_list)

    def run(self, n_jobs: int = 1, progress: bool = True, backend: str = "loky",
            chunks_per_job: int = 2, keep_selections: bool = False
            ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Execute every configuration.

        ``n_jobs > 1`` fans the grid out with joblib, in chunks grouped by
        schedule so that the per-window metric cache survives inside each
        worker.  Every worker holds its own copy of the matrices, so watch the
        memory before raising ``n_jobs``: a 5,000 x 1,000 float32 universe is
        20 MB per array per process, plus ~80 MB if the prefix-sum cache is
        built there.

        Parallelism is not free and not always a win: below a few hundred
        configurations the process startup and the pickling of the universe
        cost more than the work they distribute.  Measure before assuming.
        """
        if not self.config_list:
            raise ValueError("config_list is empty — nothing to run.")

        logger.info("Ensemble: %d configurations, %s", len(self.config_list), self.bundle)
        t0 = time.perf_counter()

        if n_jobs == 1:
            it = self.config_list
            if progress:
                it = tqdm(it, desc="Ensemble configs")
            outputs = [_run_one(self.bundle, cfg, self.periods_per_year, keep_selections)
                       for cfg in it]
        else:
            from joblib import Parallel, delayed

            chunks = _chunk_by_schedule(self.config_list, max(1, n_jobs * chunks_per_job))
            logger.info("Dispatching %d chunks over %d workers", len(chunks), n_jobs)
            batched = Parallel(n_jobs=n_jobs, backend=backend,
                               verbose=5 if progress else 0)(
                delayed(_run_chunk)(self.bundle, chunk, self.periods_per_year, keep_selections)
                for chunk in chunks
            )
            outputs = [item for batch in batched for item in batch]

        records = []
        for key, record, series, selections in outputs:
            records.append(record)
            self.oos_series[key] = series
            if selections is not None:
                self.selections[key] = selections

        self.results_df = pd.DataFrame(records).set_index("config_key")
        logger.info(
            "Ensemble finished in %.2fs (%d configs)",
            time.perf_counter() - t0, len(records),
        )
        return self.results_df, self.oos_series

    # ------------------------------------------------------------------
    def paths(self) -> pd.DataFrame:
        """All OOS paths as one frame: rows = configs, columns = timestamps."""
        from .analysis import prepare_oos_paths_dataframe

        if self.results_df is None:
            raise RuntimeError("call run() first")
        return prepare_oos_paths_dataframe(self.oos_series, self.results_df.index)

    def summary(self) -> pd.Series:
        """The headline the ensemble exists to produce: the *spread*, not a point.

        Reported on the annualized Sharpe (``oos_sharpe_ann``), using the
        ``periods_per_year`` the ensemble was built with.
        """
        if self.results_df is None:
            raise RuntimeError("call run() first")
        s = pd.to_numeric(self.results_df["oos_sharpe_ann"], errors="coerce").dropna()
        if s.empty:
            return pd.Series(dtype=float)
        return pd.Series({
            "n_configs": float(len(s)),
            "sharpe_min": float(s.min()),
            "sharpe_q25": float(s.quantile(0.25)),
            "sharpe_median": float(s.median()),
            "sharpe_mean": float(s.mean()),
            "sharpe_q75": float(s.quantile(0.75)),
            "sharpe_max": float(s.max()),
            "sharpe_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "frac_positive": float((s > 0).mean()),
            "total_run_time_sec": float(self.results_df["run_time_sec"].sum()),
        })
