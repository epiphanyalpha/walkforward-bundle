# Performance notes

> All figures below were measured on a 2-vCPU Intel Xeon @ 2.10 GHz, Python
> 3.11, numpy 2.4, pandas 3.0, numba 0.67. Reproduce with
> `python benchmarks/bench_engine.py --rows 5000 --cols 1000 --jobs 2`.
> **Two cores is a deliberately unflattering machine for the parallel numbers**
> — treat those as a floor, and the single-threaded numbers as the interesting
> ones.

## The shape of the workload

A grid sweep over a 5,000 x 1,000 universe with 576 configurations is not one
big computation. It is ~6,000 small ones, and they overlap heavily:

- Every configuration walks the same universe.
- Configurations differing only in `max_corr` or `max_columns` produce
  **identical in-sample rankings** — the parameters act after the ranking.
- On an expanding schedule, late windows are near-supersets of early ones.

So the workload is redundant before it is parallel. That ordering turned out
to matter: exploiting the redundancy paid roughly 4x, and exploiting the
parallelism paid nothing until the redundancy fix had been done properly.

## Baseline: where the time actually went

Instrumented timers around each stage of the v0.1 engine, 576 configurations,
45.3s total:

| stage | seconds | share |
|---|---|---|
| **in-sample metric** (`compute_sharpe` and friends over the full `T x k` slice) | 37.5 | **82.9%** |
| out-of-sample leg assembly (`oos[cols].mean(axis=1)` into a `pd.Series`) | 4.5 | 10.0% |
| greedy de-correlation | 1.3 | 2.9% |
| `df.loc[start:end]` slicing | 0.6 | 1.2% |
| stitching the legs | 0.4 | 1.0% |
| ranking (`argsort`) | 0.2 | 0.4% |

One line dominates: the ranking metric, recomputed over the whole window, once
per window per configuration. Everything else together is 15% of the run.

That number is the whole optimisation strategy, and it sets a hard ceiling
courtesy of Amdahl: **no amount of work on slicing, correlation or pandas
overhead can produce more than a 1.2x speedup.** The only lever that matters
is not computing the metric so many times.

## Four changes, in descending order of what they were worth

### 1. Cache the metric per window — worth 3.8x

`max_corr` and `max_columns` change what happens *after* the ranking. Two
configurations that differ only in those produce the same shortlist from the
same window, and the v0.1 engine computed it twice.

A dictionary keyed on `(metric, i0, i1, risk_free_rate, lookback)`, living on
the bundle and therefore shared across every configuration in the process,
collapses that. On the 576-config grid: **5,296 cache hits against 656 actual
evaluations** — a 9.1x reduction in metric work, from ~30 lines of code.

The arithmetic checks out end to end:

```
45.3s x (0.171 + 0.829/9.1) = 11.9s     predicted
                              12.0s     measured
```

This is the entire speedup. The three changes below are correct engineering
and were, between them, worth about 4%.

### 2. Prefix-sum moments — worth nothing here, 1.5x elsewhere

Sharpe, volatility, total return and momentum are all functions of the first
two moments, and moments are additive. Cache the prefix sums of `x` and `x²`
once, `O(T·k)`, and every window afterwards costs `O(k)`:

```
sum[i0:i1]    = cs[i1]  - cs[i0]
sumsq[i0:i1]  = cs2[i1] - cs2[i0]
var           = sumsq/n - mean²
```

Asymptotically this is the strongest optimisation in the file — it attacks the
same 83% that change #1 attacks, and attacks it harder. On the wide grid it
bought **nothing**, because #1 got there first: 656 remaining evaluations do
not amortise 80 MB of prefix sums.

It earns its place in the opposite regime — one config per schedule, so the
window cache never hits (42 configs, 7 window lengths, 3 moment-based
metrics):

| | seconds |
|---|---|
| window cache only | 3.50 |
| + prefix-sum moments | **2.32** |

Enabled by default, disabled with `FrameBundle(..., moments=False)` when
memory is tighter than time.

Two implementation notes:

- **Accumulate in float64, always.** Differencing two large prefix sums is
  where precision dies, and float32 does not have enough of it to survive.
  Even in float64 the closed form agrees with a direct `np.std` to ~1e-10
  relative, not to machine epsilon — good enough for a ranking, and pinned by
  a test so it stays that way.
- Clip the variance at zero. Cancellation produces small negative variances
  and `sqrt` of those is a `NaN` that propagates into the ranking, where the
  non-finite handling now catches it — but not computing it is better.

### 3. Standardise once, then de-correlate with dot products — worth 2.9%

The greedy de-correlation compares each candidate against everything already
kept. The naive version recomputes both columns' means and variances on every
comparison — the same column's moments, dozens of times.

Centre each column and scale it to unit L2 norm first, and a Pearson
correlation becomes a single dot product:

```
z_i = (x_i - mean_i) / ||x_i - mean_i||     ->     corr(i, j) = z_i · z_j
```

One pass to standardise, then `O(n)` per comparison with no divisions and no
reductions. The greedy loop also lost its reflected Python list in favour of a
preallocated `int64` array, which lets numba type it without boxing.

Bit-for-bit identical output, verified against the original loop across
`max_corr` and `max_columns` in
`tests/test_parity.py::test_decorrelation_matches_reference`.

Worth 2.9%. The asymptotics are genuinely better and the code is shorter, but
`top_n` is 20 — `k²` is 400, and 400 of anything is not a bottleneck. A good
optimisation applied to a stage that was never the problem.

### 4. Materialise the universe once — `FrameBundle`, worth 1.2%

`df.loc[start:end]` is a label lookup followed by a fresh block-manager
consolidation. Done once it is nothing; done once per window per config it
looked like an obvious target — 6,000 of them in a run.

`FrameBundle` stores the returns and turnover as C-order arrays plus the
index's integer representation, and turns a slice into two `searchsorted`
calls and a view. Nothing is copied.

Two traps worth naming, because both produce *silently empty* slices rather
than errors:

- `Timestamp.value` is always nanoseconds, while `DatetimeIndex.asi8` follows
  the index's own resolution — pandas ≥ 2 no longer coerces everything to ns.
  Comparing the two directly gives bounds that are off by a factor of 1,000.
- For a tz-aware index `asi8` is UTC, so a naive query timestamp has to be
  read as local-to-the-index and *then* converted.

`tests/test_parity.py::test_bundle_slice_matches_pandas_loc` pins the result
against `.loc` on both endpoints, including inclusive-end semantics.

Worth 1.2% of the runtime. It stays in because the metric cache needs
positional window bounds (`i0`, `i1`) as its key anyway, and because zero-copy
views are what let the selection stack work on slices without copying — but on
its own it was not the fix. I built it before I instrumented the stages, which
is the wrong order and produced exactly the result that mistake usually
produces.

## Parallelism, and how to get it wrong

Configurations are independent, so `joblib` should be free money. The obvious
implementation — one task per config — made the run **5.9x slower** (71s vs
12s):

- joblib pickles the universe to a worker **per task**, so 576 tasks meant 576
  round-trips of a 40 MB payload;
- each task arrived with an **empty metric cache**, so every one of the 5,296
  eliminated evaluations came straight back;
- and each rebuilt the 80 MB prefix-sum cache from scratch.

Parallelism destroyed the data locality that made the serial version fast. The
fix is to dispatch **chunks grouped by schedule** — configurations sharing
`(first_os, window_length, step_months, anchored, metric_name)` share every
cached evaluation, so keeping them in the same task keeps the cache warm.
Groups are packed into `n_jobs × chunks_per_job` bins, longest-first.

| | seconds | vs legacy |
|---|---|---|
| legacy engine | 45.7 | 1.0x |
| bundle, window cache | 12.0 | 3.8x |
| bundle, + prefix-sum moments | 12.4 | 3.7x |
| bundle, + joblib, 2 workers, chunked | 12.1 | 3.8x |
| *bundle, + joblib, 2 workers, per-config tasks* | *71.1* | *0.6x* |

On two cores, chunked parallelism recovers the overhead and gains nothing
beyond it — which is the honest result for this machine, not a disappointment.
The gain scales with core count; the *loss* from per-config dispatch scales
with grid size, which is the more useful thing to know.

**The rule of thumb:** below a few hundred configurations, run serial. Process
startup and shipping the universe cost more than the work they distribute.

## Memory

| object | 5,000 x 1,000 universe |
|---|---|
| `FrameBundle` returns + turnover (float32) | 40 MB |
| prefix-sum cache (float64, built lazily) | 80 MB |
| per-window metric cache | ~8 KB per entry per 1,000 candidates |
| **per worker process** | **~120 MB** |

`float32` for storage is deliberate: it halves the memory traffic, and the
ranking it feeds does not need more. Moment accumulation upcasts to float64
where precision is actually at risk. Pass `dtype=np.float64` to `FrameBundle`
if you need bit-exact parity with a pandas-based run.

`n_jobs=8` on this universe is ~1 GB before your own data. Check that before
raising it.

## What was not optimised, and why

- **The de-correlation is still `O(k²)` in the shortlist.** With `top_n` in
  the tens, `k²` is hundreds — the constant matters, the exponent does not.
- **The OOS leg still builds a `pd.Series` per period.** It costs ~10% and it
  is what makes the output directly usable. Kept.
- **No GPU, no threading inside the kernels.** The workload is many small
  independent problems, which is what process-level parallelism is for. A
  faster inner loop on a workload that was 90% redundant would have been the
  wrong fix.

The general lesson, paid for in this repo: three of the four changes above
were reasonable-looking optimisations aimed at stages that together accounted
for 5% of the runtime. Instrumenting the stages first would have pointed
straight at the one that mattered — and the one that mattered turned out to be
a dictionary.
