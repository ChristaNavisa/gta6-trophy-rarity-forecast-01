[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ChristaNavisa/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ChristaNavisa/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — a week after launch — as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%). Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

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
which keeps dilution (N growing faster than E) and progression (E catching
up as players play) separate instead of smearing them into one rate trend.

The history is compressed to change points (one entry per distinct page
state). On a rich history (the final case, two days of reads) each trophy's
log earners are fit against log time since release by least squares, with
the slope shrunk toward its tier median and the global median, and both
earners and players are extrapolated with damping — a log-time-linear
curve overstates saturating growth four days out. The spread combines the
house random-walk term, the fit residual, the slope standard error scaled
by the extrapolation distance, and refresh-jump volatility.

On a flat history (the practice cases, where launch night never refreshed
before the checkpoint) a generative prior is used instead: the kit's own
published record of the first refresh says tracked players went 638 ->
3,868 while total trophies earned went 943 -> 4,276. The opening trophy
(100%) accounts for 3,230 of the 3,333 new earns; the other 54 trophies
share only ~103 new earns, about 0.03 per new player, spread in
proportion to each trophy's old earners raised to 1.25 over the
non-opening trophies. The spread there is a 0.30 logit sd — round 2 tried
0.15 and the server punished it (coverage_90 0.509), because the realised
split of those 103 earns is one integer draw and deviates from any smooth
median by more than a narrow band allows.

## Local validation

Scorer checked against the worked example in RULES.txt (loss 0.403404,
exact). Truth ensemble for the practice answer (multinomial draws around
pro-rata + overdispersed Dirichlet-multinomial draws + smooth splits,
n=260, built from the kit-published refresh totals 3,868 players /
4,276 earns): this entry as shipped scores mean loss 0.117, worst 0.162,
coverage_90 0.985 (multinomial-only mean 0.104). The Bayes floor on this
problem is ~0.088 even with the true split distribution known — the
103-earn split is irreducible noise — so the practice rank this can
reach is roughly #300, not the paid top 20 (which needs ≤ ~0.022). The final case (f00, graded on the
rates of 26 Nov) is what pays; its rich history runs the trend branch,
validated on synthetic final-like worlds at loss 0.19–0.25 against a
0.32–0.48 house baseline.
