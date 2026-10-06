# Passive execution test: specification (fixed before any run)

Written and committed before the simulator was written or any passive result was seen. Nothing
below may change after the first run; any later deviation is listed in a "Deviations" section of the
results, with its reason.

## Question

Paper 3 shows that every version of the strategy loses money when it pays the spread with market
orders, and that the gross edge rises with the spread. If the edge belongs to the liquidity provider,
a strategy that posts limit orders and earns the spread should keep it. This test asks whether the
same forecasts, executed with limit orders, are profitable after the costs that remain.

## What is held fixed

* Stocks, bars, period, forecasts and target positions: exactly those of the three papers
  (`results/pertic`, `results/weights.json`). Nothing is re-estimated.
* Decisions are made at the end of 5-minute bar t, as before.

## Strategies and variants (8 configurations, all reported)

* Target positions: `cls|eq_filt` (**primary**, chosen because it has the highest breakeven in
  Paper 3), `cls|5m`, `cls|eq_sign`, `cls|lw_filt`.
* Limit price offset k, in ticks of $0.01: k = 0 (**primary**) and k = 1.

## Execution rule

* Actual position a. At the end of bar t the order is Δ = w_t − a_{t−1}. If Δ = 0 nothing is sent.
* A buy (Δ > 0) is posted at L = C_t − k × $0.01; a sell at L = C_t + k × $0.01. Size |Δ| (a
  long-to-short flip is one order of size 2).
* Fill rule, bar t+1 only: a buy fills if and only if the low of bar t+1 is strictly below L; a sell
  if and only if the high is strictly above L. The fill price is L, for the full size. A trade
  strictly through the limit means a resting order at L would have been executed before that
  trade, without assuming anything about the queue.
* Unfilled orders are cancelled at the end of bar t+1; the position stays at a_{t−1}, and the next
  decision recomputes Δ from the actual position.
* P&L of bar t+1 (bps): a_{t−1} × (ln C_{t+1} − ln C_t) + [filled] × Δ × (ln C_{t+1} − ln L).
* Positions are flat overnight. Whatever position remains at the end of a session is closed with a
  market order at the last close and pays a half-spread.

## Costs

* Filled limit orders pay no spread.
* The forced end-of-session market order pays the tick floor (**primary**), and separately the
  Abdi–Ranaldo estimate, both causal per stock and day, as in Paper 3.
* Exchange fees and rebates are excluded from the primary result. The breakeven fee is reported, in
  bps per unit traded and in dollars per share at each stock's average price.

## Consistency checks (must pass before results are read)

1. With fills forced at O_{t+1} for every order, the simulator reproduces Paper 3's next-open
   P&L exactly.
2. With every k = 0 order forced to fill at C_t, it reproduces the close-fill P&L exactly.
3. Look-ahead: simulating on data truncated at bar T gives identical P&L for every bar before T.
   Positive control: a variant that sends the order only when C_{t+1} is favourable must fail this test.

## Evaluation

* Hold-out 2022-01-03 to 2025-01-10 (the hold-out of the papers) is the primary period; the full
  period is reported alongside.
* Equal-weight book of all stocks, per-bar series; metrics as in `arxmtf/stats.py`.
* Moving-block bootstrap, 20-day blocks, 1,000 draws, 95% interval.
* Deflated Sharpe ratio with N = 110 trials (102 in the papers plus these 8).
* Reported for every configuration: fill rate, average return after filled versus unfilled orders
  (adverse selection), turnover, gross and net P&L, Sharpe, max drawdown, breakeven fee.

## Acceptance criteria for "passive execution makes the strategy profitable"

All four, for the primary configuration (`cls|eq_filt`, k = 0) in the hold-out:

* **A1.** Net Sharpe with the tick floor has a 95% bootstrap interval entirely above 0.
* **A2.** Net mean return is above 0 with the Abdi–Ranaldo cost as well.
* **A3.** Net return is positive in each of 2022, 2023 and 2024.
* **A4.** All three consistency checks pass.

If A1 to A4 hold, the conclusion is: profitable before exchange fees, with the breakeven fee
reported. If any fails, the result is reported as it is, together with the measured fill rate and
adverse selection, which say why.

## Known limits (stated now)

* Bars, not quotes: queue position, hidden liquidity and partial fills are not modelled.
* Latency is ignored: the order is posted at C_t.
* Fees and rebates depend on venue and account; hence the breakeven fee.
* One unit per stock, so capacity is not addressed.
