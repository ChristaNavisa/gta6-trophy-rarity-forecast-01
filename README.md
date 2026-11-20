[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ChristaNavisa/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ChristaNavisa/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — a week after launch — as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%). Entry for the
[make.fun bounty](https://make.fun/c/gta6-trophy-rarity-forecast-01).

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output)
```

One case per `*.in` file, one `<id>.out` per case:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.
Python 3.13 standard library only, no network, deterministic
(two runs write byte-identical output), ~0.05 s per case.

## Model

Exophase's page is a stale snapshot: between refreshes the rates and the
players-tracked count do not move at all. So the forecaster works with
counts, not rates — earners `E_i(t) = rate_i(t) * N(t) / 100` and players
`N(t)`. Earners never fall, players never fall, and the rate is the ratio,
which keeps dilution (players growing faster than earners) separate from
progression (earners catching up as players play).

- History is compressed to change points (one entry per distinct page
  state). Per trophy, log earners are fit against `u = ln(1 + hours since
  release)` by least squares, with the slope shrunk toward the tier median
  and the global median. Log players are fit the same way. Both are
  extrapolated to the target (damped ×0.85 — log-time-linear growth
  overstates a saturating curve four days out) and the ratio is the median
  rate. A trophy at 100% stays at 100%.
- Spread is a logit-scale sd combining the random-walk term
  (`0.5·√days`), the fit residual, the slope standard error scaled by the
  extrapolation distance, and refresh-jump volatility (sd of logit jumps
  seen at refreshes × expected refreshes before the target), shrunk ×0.85.
- Flat history (no refresh ever seen, as in the practice cases): the page
  is stale and a refresh before the target reveals a backlog of new
  players who have earned almost nothing beyond the opening trophy. The
  kit's published launch-night record says the first refresh took tracked
  players 638 → 3,868 while trophies earned went 943 → 4,276 — ~3,230 new
  players contributing ~103 non-opening earns (~0.03 each). Those earns
  are spread over the non-opening trophies only, weighted by old earners
  ** 1.25 (new players played less, so their earns skew to the earliest
  trophies), with a calibrated 0.15 logit spread — the refresh size is
  pinned by the published totals. Round 2 fixed two round-1 errors here:
  the old code spread the 103 over all earners including the opening
  trophy's 638, so only ~33 reached the other trophies and every median
  sat ~17% low; and its 0.35 spread floor was far wider than the refresh
  uncertainty warrants.

## Local validation

No local grade exists for this bounty (the server grades every entry), so
the practice score was estimated against proxy answers rebuilt from the
kit-published refresh totals (3,868 players, 4,276 earned at 04:30Z) under
four splits of the 103 new non-opening earns (proportional, sqrt, top-5,
uniform) plus harsher splits (all to one trophy, random Dirichlet):

| | this entry (round 2) | round 1 | house baseline |
|---|---|---|---|
| practice loss (proxy mean, 4 splits) | **0.09** | 0.12 | 1.44 |
| practice loss (proxy worst, 4 splits) | **0.22** | 0.15 | 1.59 |
| practice loss vs random integer draws (mean) | **0.09** | 0.13 | — |
| 90% coverage | 1.00 | 0.93–1.00 | 0.00 |

Round 1's server practice loss was 0.128 (rank #490), matching the
random-draw expectation above (0.13) — the truth is one integer draw of
the 103 new earns around a pro-rata split, so the honest expected loss
is ~0.09; the smooth-split numbers (~0.03–0.04) are what the same
forecast scores if the split lands exactly on its pro-rata centre. The
narrow band is punished when the split is extremely concentrated
(top-5: 0.22, all-to-one: 0.20 — worse than round 1 there), which the
proxy mean does not show.

On three synthetic final-like worlds (2 days of 10-minute reads, different
player-growth and earner-saturation shapes) the loss is 0.19–0.24 vs the
house baseline's 0.32–0.48. The true practice answers are not in the kit,
so the practice numbers above are estimates against reconstructed answers,
not measured scores.
