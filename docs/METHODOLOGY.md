# Methodology

## The problem this library exists for

A walk-forward back-test looks like the honest version of a back-test. You fit
on a window, you trade the window after it, you never let the future leak in,
and you get an out-of-sample equity curve at the end. That is real progress
over a single in-sample optimisation.

But the curve you get is a function of choices nobody in the room can justify
from first principles:

- How long is the in-sample window? 12 months? 24? 48?
- Does it expand from a fixed start, or roll?
- How often do you re-select? Quarterly? Annually?
- What do you rank candidates on — Sharpe? total return? Calmar?
- How many do you keep, and how correlated may they be?

Change any of these and you get a different curve. Present one of them and you
have, without meaning to, run a search over construction choices and reported
the winner. The out-of-sample discipline of each individual run does not
protect you from that, because the selection happened one level up, in the
choice of *which* run to show.

## The response: report the bundle

The alternative is not to pick. Run the whole grid of defensible
constructions, keep every out-of-sample path, and look at the distribution.

The output is no longer a number, it is a shape. And the shape answers a
different, better question:

> Under every reasonable way of building this walk-forward, what range of
> out-of-sample outcomes do I get?

A strategy whose paths bundle tightly across the grid has an edge that does
not depend on a lucky construction. A strategy whose paths fan out — some
excellent, some flat, some negative — has told you exactly what it is, and the
single best curve in that fan was never evidence of anything.

Both answers are useful. Only one of them is available from a single run.

## What a run does

```
candidate universe (T x k returns, optionally turnover)
        |
        v
[ WalkForwardSchedule ]  windows: (start, end), anchored or rolling
        |
        v
for each window:
    [ InitialSelector ]      rank all k candidates on the in-sample metric,
                             keep the best top_n
    [ CorrelationFilter ]    walk that shortlist best-first, drop anything
                             correlated >= max_corr with something already
                             kept; stop at max_columns
    [ min_avg_trade gate ]   optional: drop candidates whose P&L per unit of
                             turnover is below a floor
        |
        v
    [ OutOfSampleTester ]    equal-weight the survivors over the NEXT
                             step_months, record returns and turnover
        |
        v
[ FullBacktester.aggregate ] stitch the OOS legs into one continuous stream
        |
        v
[ FullBacktesterEnsemble ]   repeat for every point in the config grid
```

## The causality contract

Three properties are enforced structurally rather than by convention:

1. **The in-sample window ends before the out-of-sample period starts.** The
   OOS leg runs from `in_sample_end + 1 day` to `in_sample_end + step_months`.
   The two never share a bar.
2. **Selection uses only the window it was given.** `SelectionUnit` receives a
   slice, not the full frame. There is no path by which a later observation
   can reach a ranking.
3. **The last window with no future to trade is dropped.** The schedule stops
   `step_months` short of the end of the data, so every selection is judged on
   something.

What the library does *not* do for you: the candidate universe itself must be
causal. If the columns you feed in are P&L series from a model that was fit on
the whole history, the walk-forward is choosing among candidates that already
saw the future, and no amount of window discipline downstream repairs that.
Generate the candidate P&L causally, then bring it here.

## The two stages of selection, and why both

**Ranking** answers "which candidates were good?". Left alone, it produces a
shortlist that is usually the same trade wearing different parameter values —
ten variants of one momentum model, correlated at 0.95, whose equal-weight
portfolio is just that one model with extra steps.

**De-correlation** answers "which of these are actually different?". The
filter walks the shortlist best-first and keeps a candidate only when its
correlation with everything already selected is below `max_corr`. Applying it
*after* the ranking, rather than optimising jointly, keeps the procedure
greedy, deterministic and explainable — three properties worth more than
optimality when the object is a robustness claim.

Note the sign convention: the default test is `corr >= max_corr`, not
`abs(corr) >= max_corr`. A strongly *negatively* correlated candidate is
diversifying and is kept. Pass `use_abs_corr=True` if you want to reject on
magnitude instead.

## Choosing the grid

The grid should span choices you genuinely cannot rule out, and nothing else.
Adding axes you have no opinion about inflates the config count without
adding information; adding axes where one setting is obviously correct just
imports a known-bad configuration into your distribution.

A reasonable default for daily candidate P&L:

| axis | span | why |
|---|---|---|
| `window_length` | 12, 24, 36, 48 months | no principled answer exists |
| `step_months` | 3, 6, 12 | reselection frequency is a genuine trade-off |
| `anchored` | True, False | expanding vs rolling is a real modelling choice |
| `metric_name` | sharpe, calmar, sortino | the objective is a preference, not a fact |
| `max_corr` | 0.3, 0.5, 0.7 | how much duplication you tolerate |
| `max_columns` | 5, 10 | breadth of the deployed book |

That is 4 x 3 x 2 x 3 x 3 x 2 = 432 configurations, which is a few minutes of
compute and a distribution you can defend.

## Reading the output

`results_df` gives you one row per configuration. The temptation is to sort it
by `oos_sharpe` and look at the top. Resist it — that is the original sin in a
new costume.

Look instead at:

- **`ensemble.summary()`** — min / quartiles / median / max of OOS Sharpe
  across the grid, and the fraction of configurations that are positive. The
  median is your central estimate; the spread is your uncertainty about the
  construction; `frac_positive` is the blunt question of whether the edge
  survives choices you did not make carefully.
- **`ensemble.paths()`** — every OOS path as a row. Plot them all, faintly,
  on one chart. Tight bundling is the claim; a fan is the disclosure.
- **Sharpe as a function of one axis** (`plot_grouped_performance_boxplots`) —
  if performance collapses outside one narrow band of `window_length`, that
  parameter is not a free choice, it is a fitted one, and you should say so.
- **Clusters of paths** (`calculate_clusters`) — when the bundle separates
  into distinct groups, some construction choice is acting as a regime switch.
  Find out which one before you deploy either branch.
