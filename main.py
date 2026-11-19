#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case per *.in file (schema makefun/gta6-trophies-case/v1) from
$MAKEFUN_INPUT_DIR (else /input) and writes /output/<id>.out per case:
five quantiles (5/25/50/75/95%) of each trophy's earn rate at the target.

Model (own design, stdlib only, deterministic):
  * Exophase's page is a stale snapshot: between refreshes the rates and
    the players-tracked count do not move at all.  Work with counts, not
    rates: earners E_i(t) = rate_i(t) * N(t) / 100 and players N(t).
    Earners never fall; players never fall; the rate is the ratio, so
    dilution (N growing faster than E) and progression (E catching up as
    players play) stay separate instead of being smeared into one trend.
  * Compress the history to change points (one entry per distinct page
    state), then fit, per trophy, log earners against u = ln(1 + hours
    since release) by least squares, with the slope shrunk toward the tier
    median and the global median (small counts are noisy; shrinkage keeps
    rare trophies sane).  Fit log players the same way.  Extrapolate both
    to the target and take the ratio as the median rate.
  * Spread: a logit-scale sd combining the house random-walk term
    (0.5*sqrt(days)), the fit residual, the slope standard error scaled
    by the extrapolation distance, and refresh-jump volatility (the sd of
    logit jumps seen at refreshes, scaled by how many refreshes the
    horizon should contain).  Quantiles are the median shifted by normal
    z-scores.
  * Flat history (no refresh ever seen, e.g. the practice cases): the
    page is stale by construction and a refresh before the target will
    reveal a backlog of new players who have earned almost nothing
    beyond the opening trophy.  The kit's own published record of launch
    night says the first refresh took tracked players 638 -> 3,868 while
    total trophies earned went 943 -> 4,276, i.e. ~3,230 new players
    contributed ~103 non-opening earns (~0.03 each).  That generative
    prior is used only when the history is completely flat: new players
    arrive at that rate and their few earns spread in proportion to each
    trophy's old earners.  A trophy at 100% stays at 100%.
  * Everything is seeded-free and order-stable: two runs on the same
    case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Flat-history prior (launch-night refresh, kit-published numbers).
FLAT_PLAYER_MULT = 6.06      # 3868 / 638 tracked players at first refresh
FLAT_NEW_EARNS_PER_PLAYER = 0.032  # ~103 non-opening earns / ~3230 new players
FLAT_LOGIT_SD = 0.30         # uncertainty in the refresh size / mix

# Spread floors / caps (logit sd).
MIN_SD = 0.35
MAX_SD = 2.5
RW_SIGMA = 0.5               # house random-walk sigma per sqrt(day)
# The combined spread runs wide on rich histories (coverage ~1.0 in
# synthetic final-like worlds); shrink it modestly toward nominal.
SD_SCALE = 0.85


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit_pct(p):
    x = clamp_rate(p) / 100.0
    return math.log(x / (1.0 - x))


def inv_logit_pct(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, slope_se, n)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0, n)
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        resid_sd = math.sqrt(rss / (n - 2))
        slope_se = resid_sd / math.sqrt(sxx)
    else:
        resid_sd = 0.0
        slope_se = 0.0
    return (a, b, resid_sd, slope_se, n)


def median(vals):
    s = sorted(vals)
    m = len(s)
    if m == 0:
        return 0.0
    if m % 2:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def change_points(case, release):
    """Collapse history to one point per distinct page state.

    Returns list of dicts {u, t_hours, n, earners[list], rates[list]} in
    time order, keeping the FIRST read of each plateau (the state became
    visible then).  Also returns the list of logit jumps at refreshes.
    """
    trophies = case["trophies"]
    n_tr = len(trophies)
    points = []
    prev = None
    jumps = []
    for read in case["history"]:
        n = read.get("players_tracked")
        rates = read.get("rates") or []
        if n is None or len(rates) < n_tr:
            continue
        state = (n, tuple(rates))
        if state == prev:
            continue
        t = parse_time(read["at"])
        th = hours_between(release, t)
        u = math.log1p(max(0.0, th))
        earners = []
        for r in rates[:n_tr]:
            earners.append(None if r is None else r * n / 100.0)
        if points and prev is not None:
            old_rates = points[-1]["rates"]
            for a, b in zip(old_rates, rates[:n_tr]):
                if a is not None and b is not None and a > 0 and b > 0:
                    jumps.append(logit_pct(b) - logit_pct(a))
        points.append({"u": u, "t": th, "n": n, "earners": earners,
                       "rates": list(rates[:n_tr])})
        prev = state
    return points, jumps


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    if not case["history"]:
        # No history to learn from (should not happen): flat 1% medians
        # with a wide spread, so the case stays valid.
        rows = []
        for t in trophies:
            lg = logit_pct(1.0)
            qs = [round4(min(99.99, max(0.01, inv_logit_pct(lg + z * 1.5))))
                  for z in Z]
            rows.append({"id": t["id"], "q": qs})
        return {"trophies": rows}
    release = parse_time(case["game"]["release_date"])
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    u_target = math.log1p(max(0.0, hours_between(release, target)))

    points, jumps = change_points(case, release)
    last = points[-1]
    u_last = last["u"]
    du = max(0.0, u_target - u_last)
    n_last = last["n"]
    rates_last = last["rates"]
    earn_last = last["earners"]
    # A null in the last read means that read lacked the trophy, not that
    # its earners vanished: fall back to the last known value per trophy.
    for i in range(n_tr):
        if earn_last[i] is None:
            for p in reversed(points):
                if p["earners"][i] is not None:
                    earn_last[i] = p["earners"][i]
                    break
        if rates_last[i] is None:
            for p in reversed(points):
                if p["rates"][i] is not None:
                    rates_last[i] = p["rates"][i]
                    break

    # ---- players at target -------------------------------------------------
    n_pts = [(p["u"], math.log(max(1.0, p["n"]))) for p in points]
    flat = len(points) <= 1
    if flat:
        n_target = n_last * FLAT_PLAYER_MULT
    else:
        _, b_n, _, _, _ = ols([x for x, _ in n_pts], [y for _, y in n_pts])
        b_n = min(max(b_n, 0.0), 1.5)
        # Damp the extrapolation: launch growth decays, it does not hold
        # its log-log slope for four more days.
        log_n_target = math.log(max(1.0, n_last)) + b_n * du * 0.85
        n_target = math.exp(log_n_target)
        n_target = min(max(n_target, n_last), n_last * 25.0)

    # ---- per-trophy earners at target --------------------------------------
    # Fit log earners vs u per trophy; shrink slopes toward tier/global.
    raw_slopes = []
    fits = []
    for i in range(n_tr):
        xs, ys = [], []
        for p in points:
            e = p["earners"][i]
            if e is not None and e > 0:
                xs.append(p["u"])
                ys.append(math.log(e))
        if len(xs) >= 2:
            a, b, resid, se, n = ols(xs, ys)
        elif len(xs) == 1:
            a, b, resid, se, n = ys[0], 0.0, 0.0, 0.0, 1
        else:
            a, b, resid, se, n = 0.0, 0.0, 0.0, 0.0, 0
        raw_slopes.append(b)
        fits.append((a, b, resid, se, n))

    tier_slopes = {}
    for i, t in enumerate(trophies):
        tier_slopes.setdefault(t.get("tier", "?"), []).append(raw_slopes[i])
    tier_med = {k: median(v) for k, v in tier_slopes.items()}
    global_med = median(raw_slopes)

    # ---- refresh-jump volatility -------------------------------------------
    if jumps:
        mj = sum(jumps) / len(jumps)
        jump_sd = math.sqrt(sum((j - mj) ** 2 for j in jumps) / max(1, len(jumps) - 1)) if len(jumps) > 1 else abs(mj)
    else:
        jump_sd = 0.0
    # Expected refreshes before the target: spacing seen in history.
    if len(points) > 2:
        spans = [points[k + 1]["t"] - points[k]["t"] for k in range(len(points) - 1)]
        spacing_h = max(0.25, median(spans))
    else:
        spacing_h = 4.0
    n_refresh = max(1.0, case["horizon_minutes"] / 60.0 / spacing_h)
    shock_sd = jump_sd * math.sqrt(n_refresh) if jumps else FLAT_LOGIT_SD

    rw_sd = RW_SIGMA * math.sqrt(max(0.0, horizon_days))

    rows = []
    for i, t in enumerate(trophies):
        r_last = rates_last[i]
        if r_last is None:
            r_last = 0.0
        # Trophy everybody already holds stays held.
        if r_last >= 99.99:
            rows.append({"id": t["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_last = earn_last[i]
        if e_last is None:
            e_last = 0.0
        if flat:
            # Generative launch-night prior: new players arrive, each
            # earning ~0.03 non-opening trophies, spread pro-rata to old
            # earners; the opening trophy is handled above.
            e_total_old = sum(e for e in earn_last if e is not None)
            share = (e_last / e_total_old) if e_total_old > 0 else 0.0
            new_players = n_target - n_last
            e_new = new_players * FLAT_NEW_EARNS_PER_PLAYER * share
            e_target = e_last + e_new
            sd = math.sqrt(rw_sd ** 2 + FLAT_LOGIT_SD ** 2)
        else:
            a, b, resid, se, n_fit = fits[i]
            tier_b = tier_med.get(t.get("tier", "?"), global_med)
            w = n_fit / (n_fit + 3.0) if n_fit > 0 else 0.0
            b_shrunk = w * b + (1.0 - w) * tier_b
            b_shrunk = min(max(b_shrunk, -1.0), 2.0)
            # Damp the extrapolation like the players': log-time-linear
            # growth overstates a saturating curve four days out.
            log_e = math.log(max(0.5, e_last)) + b_shrunk * du * 0.85
            e_target = math.exp(log_e)
            e_target = min(max(e_target, e_last), n_target)
            sd = math.sqrt(rw_sd ** 2 + resid ** 2
                           + (se * du) ** 2 + shock_sd ** 2) * SD_SCALE
        sd = min(max(sd, MIN_SD), MAX_SD)
        med_rate = 100.0 * e_target / n_target if n_target > 0 else r_last
        med_rate = clamp_rate(med_rate)
        lg = logit_pct(med_rate)
        qs = []
        for z in Z:
            q = inv_logit_pct(lg + z * sd)
            qs.append(round4(min(99.99, max(0.01, q))))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
