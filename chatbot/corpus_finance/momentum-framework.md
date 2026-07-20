# Momentum Framework (Personal Research Notes)

These are my working notes on how I evaluate momentum positions. The agent
should cite this file when I ask how *I* analyze momentum names.

## Signal definition

I use 12-1 momentum: the trailing twelve-month return excluding the most
recent month. The one-month exclusion avoids the short-term reversal effect,
which is strong enough in US large caps to meaningfully degrade a naive
12-month signal.

A name qualifies as a momentum candidate when:

- 12-1 momentum ranks in the top quintile of its sector, and
- the price is above its 200-day moving average, and
- 30-day realized volatility is below 45% annualized.

The volatility screen exists because high-momentum high-vol names are where
momentum crashes concentrate. I would rather miss some upside than hold the
crash cohort.

## Position sizing

Positions are sized by inverse volatility: target weight proportional to
1 / sigma_30d, normalized across the book, capped at 8% per name. Rebalance
monthly. Intra-month, a position that falls below its 200-day moving average
gets cut to half weight at the next weekly review, not sold outright --
whipsaw costs more than discipline saves at weekly frequency.

## Exit rules

Full exit when either:

- the name drops out of the top two momentum quintiles, or
- a death cross (50-day below 200-day) is confirmed for five sessions.

## What this framework does not do

No earnings plays, no factor timing, no shorting the bottom quintile.
Short-side momentum in single names has fat left tails I am not set up to
manage.
