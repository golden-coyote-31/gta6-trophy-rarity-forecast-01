# GTA VI Trophy Rarity Forecaster

Entry for the make.fun competition **gta6-trophy-rarity-forecast-01**:
forecast, for every PS5 trophy of Grand Theft Auto VI, the percent of
Exophase-tracked players who will have earned it at the case target, as
five quantiles (5/25/50/75/95%), scored by pinball loss on the logit
scale.

## Running it

`main.py` (Python 3.13, standard library only) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` files to
`$MAKEFUN_OUTPUT_DIR` (default `/output`):

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}]}
```

The program is a pure function of the case file: no randomness, no clock,
no network, so repeated runs are byte-identical.

## Method

The house's read history is a step function — the Exophase page only
changes when its counts refresh — so the history is first collapsed to
its distinct page states. A trophy's earn rate is earners over players,
r = E / N, and the two sides move for different reasons, so the forecast
works on both scales against u = ln(1 + hours since release):

- a direct fit: logit(r) linear in u over the whole history, weighted
  least squares with a short recency half-life and count-based weights;
- a counts fit: ln(E + 0.5) linear in u, carried to the target by the
  player pool's own fitted growth line and divided back;
- a local fit: logit(r) linear in u over the last 30 hours of states
  only, which hears recent regime changes the global fit smooths over.

Slopes are shrunk toward the mean slope of each trophy's rate band.
Every fit's move from the last observed rate is damped (growth bends
toward its ceiling: slopes cross the horizon gap with an exponential
decay, and the total move is soft-capped), the median blends the three
damped predictions in logit space with soft week-one ceilings by trophy
tier, and a launch-window dilution prior applies while the tracked pool
is still multiplying faster than anyone earns. The spread combines the
fit residual scale, extrapolation leverage, the observed refresh step
size, the disagreement between the fits, and a sqrt-horizon term.

Validation: on synthetic final-like scenarios (cohort-growth worlds,
saturating logit-curve worlds, and worlds anchored at the real
launch-night page, all sampled through an Exophase-style refresh clock),
the forecaster's mean pinball loss is ≈ 0.20, against ≈ 0.27 for a
last-24-hours curve-fit baseline and ≈ 0.37 for persistence — the two
strategies the house entries are built from.
