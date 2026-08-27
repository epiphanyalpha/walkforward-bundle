"""
Zero-copy slicing layer.

The walk-forward loop asks for the same DataFrame over and over, with slightly
different date boundaries.  Doing that with ``df.loc[start:end]`` costs a
label lookup plus a fresh block-manager consolidation *per slice per config* —
which, on a grid of a few hundred configurations, dominates the runtime and
never touches the actual maths.

``FrameBundle`` materializes the returns / turnover matrices once, as C-order
NumPy arrays, and turns every slice into a pair of ``searchsorted`` calls plus
a view.  Nothing is copied, nothing is re-indexed, and the selection code
downstream sees an object that quacks like a DataFrame (``.values``,
``.columns``, ``.index``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["ArrayFrame", "FrameBundle"]


@dataclass(frozen=True)
class ArrayFrame:
    """A minimal, read-only stand-in for a DataFrame slice.

    Exposes exactly the three attributes the selection stack needs, so the
    same code path works whether it is handed a real ``pd.DataFrame`` or one
    of these views.
    """

    values: np.ndarray
    columns: pd.Index
    index: pd.Index

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    def __len__(self) -> int:
        return self.values.shape[0]

    @property
    def empty(self) -> bool:
        return self.values.shape[0] == 0

    def to_frame(self) -> pd.DataFrame:
        """Materialize as a real DataFrame (copies)."""
        return pd.DataFrame(self.values, index=self.index, columns=self.columns)


class FrameBundle:
    """Returns + (optional) turnover, pre-materialized and column-aligned.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide frame, rows = timestamps, columns = candidate strategies/assets.
    turnover : pd.DataFrame, optional
        Same shape / same columns as ``returns``.
    dtype : np.dtype
        Storage dtype.  ``float32`` halves the memory traffic and is plenty
        for return series; use ``float64`` if you need bit-exact parity with
        a pandas-based run.
    validate : bool
        Check that the index is monotonic and free of duplicates. Turn it off
        only if you have already guaranteed both.
    """

    __slots__ = ("index", "columns", "returns", "turnover", "_index_i8", "_unit",
                 "_tz", "_metrics", "_use_moments", "bars_per_year")

    def __init__(
        self,
        returns: pd.DataFrame,
        turnover: Optional[pd.DataFrame] = None,
        dtype: np.dtype = np.float32,
        validate: bool = True,
        moments: bool = True,
    ) -> None:
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("returns must be indexed by a DatetimeIndex")

        if validate:
            if not returns.index.is_monotonic_increasing:
                returns = returns.sort_index()
                if turnover is not None:
                    turnover = turnover.sort_index()
            if returns.index.has_duplicates:
                keep = ~returns.index.duplicated(keep="first")
                returns = returns.loc[keep]
                if turnover is not None:
                    turnover = turnover.loc[keep]

        self.index = returns.index
        self.columns = returns.columns
        # ``asi8`` is expressed in the index's own resolution, which pandas >= 2
        # no longer forces to nanoseconds. Query timestamps must be converted to
        # that same unit before they can be compared against it.
        self._unit = getattr(self.index, "unit", "ns")
        self._tz = self.index.tz
        self._index_i8 = np.ascontiguousarray(self.index.asi8)
        self.returns = np.ascontiguousarray(returns.to_numpy(dtype=dtype))

        if turnover is not None:
            turnover = turnover.reindex(index=self.index, columns=self.columns)
            self.turnover = np.ascontiguousarray(turnover.to_numpy(dtype=dtype))
        else:
            self.turnover = None

        self._use_moments = moments
        self._metrics = None
        self.bars_per_year = self._infer_bars_per_year()

    def _infer_bars_per_year(self) -> float:
        """Bars per year, measured as bars observed over the span covered.

        This is what lets a configuration speak in months — the unit a human
        reasons in — while the kernels receive bar counts, the unit they
        operate in. Without it the two silently disagree.
        """
        if len(self.index) < 3:
            return 252.0
        unit_per_day = {"ns": 86_400e9, "us": 86_400e6, "ms": 86_400e3, "s": 86_400.0}
        per_day = unit_per_day.get(self._unit, 86_400e9)
        span_days = float(self._index_i8[-1] - self._index_i8[0]) / per_day
        if span_days <= 0:
            return 252.0
        # observed rate over the whole span, not the median gap: a
        # business-day index has a median gap of one day but only ~252 bars a
        # year, and weekends are exactly the kind of gap a median hides.
        return (len(self.index) - 1) * 365.25 / span_days

    def bars_for_months(self, months: float) -> int:
        """Convert a calendar span to a bar count on this index."""
        return max(1, int(round(months * self.bars_per_year / 12.0)))

    # ------------------------------------------------------------------
    @property
    def metrics(self):
        """Shared per-window metric cache (see :mod:`validation.moments`)."""
        if self._metrics is None:
            from .moments import MetricEngine

            self._metrics = MetricEngine(self, moments=self._use_moments)
        return self._metrics

    def __getstate__(self):
        """Do not ship the caches across process boundaries.

        joblib pickles the bundle to every worker; the prefix-sum cache is
        derived data and re-deriving it there is far cheaper than sending it.
        """
        state = {k: getattr(self, k) for k in self.__slots__}
        state["_metrics"] = None
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    def bounds(self, start, end, inclusive_end: bool = True) -> Tuple[int, int]:
        """Positional half-open bounds ``[i0, i1)`` matching ``df.loc[start:end]``.

        ``.loc`` slicing on a sorted DatetimeIndex is inclusive on both ends,
        hence ``side='right'`` for the upper bound.
        """
        lo = self._to_i8(start)
        hi = self._to_i8(end)
        i0 = int(np.searchsorted(self._index_i8, lo, side="left"))
        i1 = int(np.searchsorted(self._index_i8, hi, side="right" if inclusive_end else "left"))
        return i0, max(i0, i1)

    def _to_i8(self, ts) -> np.int64:
        """Timestamp -> integer on the same epoch scale as ``index.asi8``.

        Two traps live here.  ``Timestamp.value`` is always nanoseconds, while
        ``asi8`` follows the index's own resolution (pandas >= 2 no longer
        coerces everything to ns).  And for a tz-aware index ``asi8`` is UTC,
        so a naive query timestamp has to be read as local-to-the-index first
        and then converted.
        """
        t = pd.Timestamp(ts)
        if self._tz is not None:
            t = t.tz_localize(self._tz) if t.tzinfo is None else t.tz_convert(self._tz)
            t = t.tz_convert("UTC").tz_localize(None)
        elif t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return np.int64(
            np.datetime64(t.to_datetime64()).astype(f"datetime64[{self._unit}]").astype("int64")
        )

    def slice(self, start, end, inclusive_end: bool = True):
        """Return ``(returns_view, turnover_view_or_None)`` as :class:`ArrayFrame`."""
        i0, i1 = self.bounds(start, end, inclusive_end=inclusive_end)
        return self.islice(i0, i1)

    def islice(self, i0: int, i1: int):
        idx = self.index[i0:i1]
        ret = ArrayFrame(self.returns[i0:i1], self.columns, idx)
        tur = (
            ArrayFrame(self.turnover[i0:i1], self.columns, idx)
            if self.turnover is not None
            else None
        )
        return ret, tur

    # ------------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        n = self.returns.nbytes
        if self.turnover is not None:
            n += self.turnover.nbytes
        return n

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        t, k = self.returns.shape
        return (
            f"FrameBundle({t} rows x {k} cols, dtype={self.returns.dtype}, "
            f"turnover={'yes' if self.turnover is not None else 'no'}, "
            f"{self.nbytes / 1e6:.1f} MB)"
        )
