"""
walkforward-bundle — ensemble walk-forward backtesting for systematic trading.

A single walk-forward back-test produces one equity curve, and that curve is
as much a statement about the window length you happened to pick as it is
about the strategy.  This library runs the whole grid of reasonable
walk-forward constructions and hands you the *bundle* of out-of-sample paths,
so the question becomes "how tightly do these cluster?" rather than "how good
is this one line?".
"""
from __future__ import annotations

__version__ = "0.2.0"

from ._frames import ArrayFrame, FrameBundle
from .correlation_filter import CorrelationFilter, get_uncorrelated_indices
from .ensemble_backtester import (
    FullBacktesterEnsemble,
    config_key,
    generate_config_list,
)
from .full_backtester import FullBacktester
from .initial_selector import InitialSelector
from .metrics import METRICS
from .oos_tester import OutOfSampleTester
from .selection_unit import SelectionUnit
from .walkforward import WalkForwardRunner, WalkForwardSchedule

__all__ = [
    "__version__",
    # core
    "FullBacktesterEnsemble",
    "generate_config_list",
    "config_key",
    "FullBacktester",
    "METRICS",
    # building blocks
    "FrameBundle",
    "ArrayFrame",
    "WalkForwardSchedule",
    "WalkForwardRunner",
    "SelectionUnit",
    "InitialSelector",
    "CorrelationFilter",
    "get_uncorrelated_indices",
    "OutOfSampleTester",
]


def __getattr__(name):
    """Analysis and plotting live behind optional dependencies.

    ``pip install walkforward-bundle`` gives you the engine with numpy, pandas
    and numba only.  scikit-learn / matplotlib / seaborn are pulled in by the
    ``[analysis]`` extra, and importing them lazily keeps the core import
    cheap and the dependency footprint honest.
    """
    analysis_exports = {
        "prepare_oos_paths_dataframe",
        "prepare_oos_paths_dataframe_2",
        "calculate_clusters",
    }
    viz_exports = {
        "plot_grouped_equity_curves",
        "plot_cluster_centroids",
        "plot_grouped_performance_boxplots",
        "plot_config_space_pca",
        "plot_sharpe_heatmap",
        "print_top_clusters",
        "plot_config_space_umap",
    }
    if name in analysis_exports:
        from . import analysis
        return getattr(analysis, name)
    if name in viz_exports:
        from . import visualization
        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
