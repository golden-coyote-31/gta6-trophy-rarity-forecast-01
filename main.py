"""gta6-trophy-rarity-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, the percent of the
players Exophase tracks who will have earned it at the case's target, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95), scored by pinball loss on the logit
scale.

Method. A case's history is the house's reads of the Exophase page. The page
only changes when Exophase refreshes its counts, so the history is a step
function: first collapse it to its distinct page states. A trophy's earn rate
is earners over players tracked, r = E / N, and the two sides move for
different reasons: N grows as more players sync a profile (fast and
decelerating after a launch), E grows as the tracked players reach the
trophy. The forecast therefore works on both scales at once, against
u = ln(1 + hours since release):

  - a direct fit: logit(r) linear in u, weighted least squares over the
    distinct states, with a short recency half-life (the regime near the
    checkpoint is the one that survives to the target) and better-counted
    states weighing more;
  - a counts fit: ln(E + 0.5) linear in u, with ln(N) fitted the same way,
    read at the target and divided back, E capped just under N;
  - a local fit: logit(r) linear in u over just the last 30 hours of
    states, the only voice that hears a recent regime change;
  - per-trophy slopes are shrunk toward the mean slope of the trophy's
    rate band (trophies sitting at similar rates grow alike), which keeps
    a trophy with a jumpy short history from extrapolating wildly;
  - every fit's move from the last rate is damped: growth bends toward
    its ceiling, so a slope is carried across the horizon gap with an
    exponential decay and the total move is soft-capped;
  - the median blends the three damped predictions in logit space (the
    counts fit leads for rare trophies, where the logit of a rounded
    rate is noisy), nudged a tenth of the way back to the last rate, and
    pulled softly toward week-one ceilings by tier (a platinum cannot
    sit at 60% a week in);
  - when the history holds almost no dynamics (a couple of states, a game
    hours old), no slope is identifiable and a launch prior applies
    instead: in the first hours the tracked pool grows much faster than
    anyone earns, so rates dilute; the median shifts down with the horizon
    and the spread widens;
  - the spread combines the fit's residual scale (shrunk to the pooled
    one), the extrapolation leverage at the target, the typical size of a
    refresh step in the history, the disagreement between the fits, and a
    term growing with the square root of the horizon in days, floored so
    no trophy claims false precision.

Everything is a pure function of the case file: no randomness, no clock, no
network, so two runs write the same bytes.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. Reads every <id>.in in the input
directory (MAKEFUN_INPUT_DIR or /input), writes <id>.out to the output
directory (MAKEFUN_OUTPUT_DIR or /output):
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (we ship none; kept for parity).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each scored level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Soft week-one ceilings by tier, in percent: a forecast median above the
# cap is pulled back to the cap plus 30% of the excess (in logit space).
TIER_CAP = {"platinum": 15.0, "gold": 55.0, "silver": 75.0, "bronze": 97.0}
# Recency half-life for fit weights, in hours of read time. Short on
# purpose: a launch's growth regime changes by the day, and the slope that
# matters is the one near the checkpoint, not the launch-night explosion.
# (Constants in this block were tuned on synthetic launch scenarios
# spanning cohort growth, saturating logit curves and paths anchored at
# the real launch-night page.)
HALF_LIFE_H = 8.0
# Slope shrinkage strength (pseudo-count of band evidence).
SHRINK_K = 2.5
# Spread: coefficient of the sqrt-horizon (days) term, in logit units.
HORIZON_K = 0.10
# Growth decays as a trophy's curve bends toward its ceiling: the fitted
# slope is extrapolated with this decay rate per unit of u, so the move
# from the last state to the target is slope * (1 - exp(-DECAY*du)) / DECAY
# rather than slope * du.
DECAY = 0.7
# Soft cap on the median's move from the last rate, in logit units.
MAX_MOVE = 1.8


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_anchor(case: dict, first_read: datetime) -> datetime:
    """The instant u is measured from: the release date at 00:00 UTC."""
    raw = (case.get("game") or {}).get("release_date") or ""
    try:
        d = datetime.fromisoformat(raw)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except ValueError:
        return first_read


def u_of(when: datetime, anchor: datetime) -> float:
    hours = (when - anchor).total_seconds() / 3600.0
    return math.log(1.0 + max(hours, -0.95))


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x.

    Returns (a, b, xbar, sxx, sw): the intercept, slope, weighted mean of x,
    weighted sum of (x - xbar)^2 and the total weight, or None when the fit
    is not identifiable.
    """
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (ybar, 0.0, xbar, sxx, sw)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (ybar - b * xbar, b, xbar, sxx, sw)


def distinct_states(history):
    """Collapse consecutive identical page reads to (time, players, rates).

    Exophase refreshes on its own clock; between refreshes every read shows
    the same page, so only the changes carry information.
    """
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key != prev_key:
            states.append((parse_time(read["at"]),
                           read.get("players_tracked") or 0,
                           list(rates)))
            prev_key = key
    return states


def band_of(rate: float) -> int:
    if rate < 0.5:
        return 0
    if rate < 2.0:
        return 1
    if rate < 10.0:
        return 2
    if rate < 40.0:
        return 3
    return 4


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    horizon_h = horizon_days * 24.0

    states = distinct_states(history)
    anchor = release_anchor(case, states[0][0])
    us = [u_of(t, anchor) for t, _, _ in states]
    u_last = us[-1]
    u_target = u_of(target, anchor)
    age_h = (checkpoint - anchor).total_seconds() / 3600.0
    last_time = states[-1][0]
    # Recency weights over states (the last state weighs 1).
    rec = [0.5 ** (((last_time - t).total_seconds() / 3600.0) / HALF_LIFE_H)
           for t, _, _ in states]

    # ---- players tracked: ln N linear in u -------------------------------
    n_fit = None
    if len(states) >= 2 and all(p > 0 for _, p, _ in states):
        n_fit = wls(us, [math.log(p) for _, p, _ in states], rec)
    n_last = states[-1][1] or 1
    du_full = max(u_target - u_last, 0.0)
    # The fitted slope's move over the horizon, bent by the decay: growth
    # in ln time slows as the pool matures, so a slope is never carried
    # across the whole gap at full strength.
    decayed_du = ((1.0 - math.exp(-DECAY * du_full)) / DECAY
                  if du_full > 0 else 0.0)
    if n_fit is not None:
        ln_n_target = math.log(n_last) + n_fit[1] * decayed_du
        n_target = math.exp(min(max(ln_n_target, math.log(n_last * 0.9)),
                                math.log(n_last * 60.0)))
    else:
        n_target = float(n_last)

    # ---- per-trophy first pass: both fits, unshrunk ----------------------
    fits = []
    jumps = []
    for i in range(n):
        obs = []  # (u, rate, players, recency)
        for s, (t, players, rates) in enumerate(states):
            r = rates[i] if i < len(rates) else None
            if r is not None:
                obs.append((us[s], float(r), players, rec[s]))
        last_rate = obs[-1][1] if obs else None
        fit = {"obs": obs, "last": last_rate, "band": band_of(last_rate or 0.0)}
        if last_rate is not None:
            for s in range(1, len(obs)):
                if obs[s - 1][1] > 0 and obs[s][1] > 0:
                    jumps.append(abs(logit_pct(obs[s][1])
                                     - logit_pct(obs[s - 1][1])))
        if len(obs) >= 2:
            ws_a = [w * (0.3 + 0.7 * min((r * p / 100.0) / 25.0, 1.0))
                    for _, r, p, w in obs]
            fa = wls([o[0] for o in obs],
                     [logit_pct(o[1]) for o in obs], ws_a)
            fb = wls([o[0] for o in obs],
                     [math.log(o[1] * o[2] / 100.0 + 0.5) for o in obs], ws_a)
            fit["a"] = fa
            fit["b"] = fb
            # Local direct fit: the last 30 hours of states only. A launch's
            # regime bends fast; the recent slope is the one that survives
            # to the target, and when the recent page has gone flat this
            # fit says so, however steep the launch-night climb was.
            u_cut = u_of(last_time - timedelta(hours=30), anchor)
            loc = [(o, w) for o, w in zip(obs, ws_a) if o[0] >= u_cut]
            if len(loc) >= 3:
                fit["l"] = wls([o[0] for o, _ in loc],
                               [logit_pct(o[1]) for o, _ in loc],
                               [w for _, w in loc])
            if fa is not None:
                resid = [logit_pct(o[1]) - (fa[0] + fa[1] * o[0])
                         for o in obs]
                fit["resid"] = math.sqrt(
                    sum(w * e * e for e, w in zip(resid, ws_a))
                    / max(sum(ws_a), 1e-9))
        fits.append(fit)

    # Pooled scales: residual (median over trophies) and refresh step size.
    resids = sorted(f["resid"] for f in fits if f.get("resid") is not None)
    pooled_resid = resids[len(resids) // 2] if resids else 0.35
    jumps.sort()
    pooled_step = jumps[len(jumps) // 2] if jumps else 0.5

    # Band mean slopes for the fits (weight-averaged over fitted trophies).
    band_slope = {}
    for f in fits:
        for key in ("a", "b", "l"):
            f.setdefault(key, None)
    for band in range(5):
        for key in ("a", "b", "l"):
            num = den = 0.0
            for f in fits:
                if f["band"] == band and f.get(key) is not None:
                    wgt = f[key][4]
                    num += wgt * f[key][1]
                    den += wgt
            band_slope[(band, key)] = (num / den) if den > 0 else None
    all_slope = {}
    for key in ("a", "b", "l"):
        num = den = 0.0
        for f in fits:
            if f.get(key) is not None:
                num += f[key][4] * f[key][1]
                den += f[key][4]
        all_slope[key] = (num / den) if den > 0 else 0.0

    def shrunk_pred(fit, key, fallback_logit):
        """Slope-shrunk prediction of fit `key` at the target, in its units."""
        f = fit.get(key)
        if f is None:
            return None
        a, b, xbar, sxx, sw = f
        band_mean = band_slope.get((fit["band"], key))
        if band_mean is None:
            band_mean = all_slope[key]
        b_shrunk = (sw * b + SHRINK_K * band_mean) / (sw + SHRINK_K)
        # Re-anchor the intercept so the line still passes through the
        # weighted centroid with the shrunk slope.
        ybar = a + b * xbar
        return (ybar - b_shrunk * xbar) + b_shrunk * u_target, b_shrunk, xbar, sxx, sw

    # Launch prior regime: almost no dynamics observed in a young game.
    few_states = len(states) <= 3
    if few_states and age_h <= 14.0:
        prior_strength = 1.0
    elif few_states and age_h <= 48.0:
        prior_strength = 0.5
    else:
        prior_strength = 0.0
    dilution = prior_strength * min(1.5 * math.log(1.0 + horizon_h / 2.0), 1.3)

    rows = []
    for i, trophy in enumerate(trophies):
        fit = fits[i]
        last = fit["last"]
        if last is None:
            # Never seen: a rare trophy nobody has earned yet in any read.
            med = logit_pct(0.05)
            sigma = 1.6
        elif last >= 99.9:
            # Earned by effectively every tracked player (it survived a
            # sixfold jump in players tracked without moving); it stays
            # pinned to the top of the scale.
            med = logit_pct(99.99)
            sigma = 0.10
        else:
            pa = shrunk_pred(fit, "a", None)
            pb = shrunk_pred(fit, "b", None)
            last_lg = logit_pct(last)
            # Damp each fit's move from the last rate: fitted slopes come
            # from a bending curve, so the move is scaled by the decayed
            # share of the gap and soft-capped.
            frac = (decayed_du / du_full) if du_full > 0 else 1.0

            def damp(pred_logit):
                move = (pred_logit - last_lg) * frac
                if abs(move) > MAX_MOVE:
                    move = math.copysign(
                        MAX_MOVE + 0.3 * (abs(move) - MAX_MOVE), move)
                return last_lg + move

            pred_a = damp(pa[0]) if pa is not None else None
            pred_b = None
            if pb is not None:
                e_target = math.exp(pb[0]) - 0.5
                e_target = min(max(e_target, 0.0), 0.99995 * n_target)
                pred_b = damp(logit_pct(100.0 * e_target / n_target))
            pl = shrunk_pred(fit, "l", None)
            pred_l = damp(pl[0]) if pl is not None else None
            if pred_a is not None and pred_b is not None:
                gap = abs(pred_b - pred_a)
                if gap > 1.2:
                    pred_b = pred_a + math.copysign(
                        1.2 + 0.25 * (gap - 1.2), pred_b - pred_a)
            # The counts fit earns its weight where rates are small and
            # logit-of-rounded-rates is noisy; the direct fit leads once a
            # trophy is common. The local fit, when it exists, takes a
            # fixed share: it is the only voice that hears the recent
            # regime change.
            w_b = 0.55 if last < 2.0 else (0.4 if last < 10.0 else 0.25)
            parts = []
            if pred_a is not None:
                parts.append((pred_a, 1.0 - w_b))
            if pred_b is not None:
                parts.append((pred_b, w_b))
            if parts:
                med = (sum(v * w for v, w in parts)
                       / sum(w for _, w in parts))
            else:
                med = last_lg
            if pred_l is not None and parts:
                med = 0.65 * med + 0.35 * pred_l
            elif pred_l is not None:
                med = pred_l
            live = [p for p in (pred_a, pred_b, pred_l) if p is not None]
            disagree = (max(live) - min(live)) if len(live) >= 2 else 0.0
            med = 0.9 * med + 0.1 * logit_pct(last)
            # Soft tier ceiling.
            cap = TIER_CAP.get(trophy.get("tier"), 97.0)
            cap_lg = logit_pct(cap)
            if med > cap_lg:
                med = cap_lg + 0.3 * (med - cap_lg)
            med -= dilution
            # ---- spread ----
            resid_i = fit.get("resid")
            nobs = len(fit["obs"])
            if resid_i is None:
                s_i = pooled_resid
            else:
                s_i = math.sqrt((nobs * resid_i ** 2
                                 + 4.0 * pooled_resid ** 2) / (nobs + 4.0))
            lev = 0.0
            if pa is not None:
                # pa is (prediction, shrunk slope, xbar, sxx, total weight):
                # the weighted-LS leverage of a prediction at the target.
                _, _, xbar, sxx, sw = pa
                lev = (u_target - xbar) ** 2 / sxx if sxx > 1e-12 else 1.0
                lev += 1.0 / max(sw, 1.0)
            sigma = math.sqrt(s_i ** 2 * (1.0 + lev)
                              + (0.55 * pooled_step) ** 2
                              + (0.35 * disagree) ** 2
                              + (HORIZON_K * math.sqrt(horizon_days)) ** 2)
            floor = 0.38 if horizon_days >= 1.0 else 0.30
            if prior_strength > 0.0:
                floor = max(floor, 0.9)
            if last <= 0.0:
                # Nobody tracked has earned it yet — but "yet" is the
                # operative word a week into a launch. The rounded 0.00
                # only says the count is below the page's resolution; the
                # first earners can appear any day, so the median sits at
                # a small positive rate with a wide spread rather than
                # claiming a precise zero.
                med = max(med, logit_pct(0.05))
                sigma = max(sigma * 1.35, 1.3)
            sigma = min(max(sigma, floor), 2.4)
        quantiles = []
        prev = 0.0
        for z in Z:
            v = round4(min(99.99, max(0.01, pct_from_logit(med + z * sigma))))
            if v < prev:
                v = prev
            quantiles.append(v)
            prev = v
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
