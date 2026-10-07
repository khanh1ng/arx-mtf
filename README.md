<!-- TOP:START -->
# arx-mtf

**A real but untradable intraday edge: multi-timeframe direction forecasting on 431 S&P 500
stocks.** Three papers and a follow-up execution test on 5-minute bars; evaluation 2020-12-18 to
2025-01-10 (4.05 years). Every number below and in the papers is generated from `results/`.

**Summary.** A walk-forward ARX model forecasts the next 5-minute return at six horizons, 5 minutes
to 4 hours. Before costs the forecast has a statistically significant edge: hit rate 0.506 and
gross Sharpe 4.99 (95% CI [4.10, 5.97]) with next-open fills, 4.65 [3.77, 5.82] in the
2022–2025 hold-out. The edge is a one-bar reversal, most of it bid–ask bounce. It breaks even at
0.10–0.22 bps per unit traded, while the cheapest possible market order pays 0.47 bps at the
median stock, and resting limit orders are adversely selected. The contribution is a measured
account of where an intraday edge comes from and what it would cost to capture it.

## Hypotheses and verdicts

| | Hypothesis | Test | Evidence | Verdict |
|---|---|---|---|---|
| H1 | Next-bar returns are forecastable out of sample | Walk-forward ARX, next-open fills; moving-block bootstrap; deflated Sharpe over every configuration run | Hit rate 0.506; gross Sharpe 4.99 [4.10, 5.97], hold-out 4.65 [3.77, 5.82]; best of 102 configurations: deflated Sharpe probability 0.978 | **Supported** |
| H2 | Bar-level order flow adds information | Flow coefficients across 17,240 refits; proxy replaced by a plain sign | Flow coefficients positive in 48%–56% of refits (no stable sign); only the latest return is stable, negative in 76% | **Not supported** |
| H3 | Averaging horizons keeps the edge at lower turnover | Equal-weight average of six standardised forecasts against the 5-minute model | Keeps 58% of gross return at 33% of turnover; breakeven 0.103 → 0.182 bps. Filtered books: hold-out Sharpe CIs [−0.22, 2.02] and [−0.40, 1.97] include zero | **Supported** for equal weights; **not significant** for filtered books |
| H4 | A close-fill backtest measures a tradable edge | Proposition 1: close fill minus next-open fill = position change × jump between the two prints | The jump is bid–ask bounce: 46% of the close-fill edge; Sharpe 9.01 → 4.99 | **Rejected** (artifact, proved) |
| H5 | The edge survives market-order costs | Causal per-stock, per-day costs; exact tick floor as lower bound | Breakeven 0.10–0.22 bps against at least 0.47 bps; all 46 versions lose at the tick floor | **Rejected** |
| H6 | Limit orders collect the spread instead | Passive test with a specification frozen before the run; trade-through fills | 83% of orders fill; price moves −4.6 bps after fills, +25.3 after misses; net Sharpe −15.1 [−17.1, −13.4]; all 8 configurations worse than market orders | **Rejected** (adverse selection) |
| H7 | A Hasbrouck VAR on bars measures information in trades | Closed-form identities against the estimates | A regression-free formula matches the impact coefficient (correlation 0.9986); one control removes 97% of the lagged impact | **Rejected** (mostly identity) |

## Results before costs

| Strategy | Period | $1M grows to | CAGR | Sharpe [95% CI] | Max drawdown | Months positive | Breakeven (bps) |
|---|---|---:|---:|---:|---:|---:|---:|
| 5-minute model | Hold-out 2022–2025, 3.01 yrs | $1,756,488 | +20.6% | 4.65 [3.77, 5.82] | −2.6% | 92% | 0.10 |
| 5-minute model | Full, 4.05 yrs | $2,141,607 | +20.7% | 4.99 [4.10, 5.97] | −2.6% | 92% | 0.10 |
| Six timeframes, equal weights | Hold-out 2022–2025, 3.01 yrs | $1,349,527 | +10.5% | 3.18 [2.28, 4.44] | −2.3% | 84% | 0.16 |
| Six timeframes, equal weights | Full, 4.05 yrs | $1,552,316 | +11.5% | 3.61 [2.66, 4.66] | −2.3% | 84% | 0.18 |
| Six timeframes, equal weights, confidence filter | Hold-out 2022–2025, 3.01 yrs | $1,073,071 | +2.4% | 0.80 [−0.22, 2.02] | −4.1% | 62% | 0.17 |
| Six timeframes, equal weights, confidence filter | Full, 4.05 yrs | $1,160,997 | +3.8% | 1.32 [0.32, 2.39] | −4.1% | 62% | 0.28 |
| Six timeframes, learned weights, confidence filter | Hold-out 2022–2025, 3.01 yrs | $1,069,992 | +2.3% | 0.81 [−0.40, 1.97] | −3.8% | 68% | 0.22 |
| Six timeframes, learned weights, confidence filter | Full, 4.05 yrs, weights learned here | $1,127,976 | +3.0% | 1.12 [0.13, 2.16] | −3.8% | 68% | 0.31 |

Next-open fills, the first price available after each decision; equal-weight book of 431 stocks;
positions closed at every session end. **Net of the smallest possible cost every version loses
money**; the best hold-out version turns $1M into $870,953.

## Mechanism: why the edge exists and why it cannot be captured

* **The Sharpe ratio comes from breadth, not accuracy.** The information coefficient is 0.0108 and a
  single stock's Sharpe ratio is 0.78. With average pairwise correlation 0.025 the book holds
  about 36 independent bets, so the fundamental law predicts 4.70; the realised daily Sharpe
  ratio is 4.76.
* **The reversal is the spread.** After a down bar, the jump from the close to the next open reverses the move
  in 94% of stocks.
  Gross breakeven rises from 0.066 to 0.197 bps from the tightest to the widest spread quintile
  and correlates 0.63 with each stock's spread.
* **Filled limit orders are the losing ones.** A buy resting at the last close fills only if the price
  falls through it, which is when the reversal forecast is wrong. Breaking even would need a rebate of
  3.47 bps per unit traded.

## Threats to validity

| Threat | How it is handled | What remains |
|---|---|---|
| Look-ahead | Truncation-invariance tests on signals (18 of 18 cuts) and on execution (18 cuts), each with a positive control (signals: caught; execution: 18 of 18) | None known |
| Data snooping | Deflated Sharpe over 102 configurations; hold-out 2022–2025; timeframe weights learned on 2021 only | Configurations were chosen after earlier results on the same data |
| Fill assumption | Three fill rules; next-open is primary; Proposition 1 quantifies the gap | None |
| Costs | Exact tick-floor lower bound plus Abdi–Ranaldo and Corwin–Schultz, all causal | Bars, not quotes; no market impact, so the negative verdict is conservative |
| Passive fills | Trade-through rule, no queue assumption | Queue position, hidden liquidity and latency not modelled |
| Survivorship | Not handled | Stocks with data through 2025 only |

## Implications

* **For research:** judge a high-turnover strategy by its breakeven cost, not its Sharpe ratio; fill at
  the first print after the signal; and check any bar-level regression for construction identities.
* **For trading:** not tradable as specified. A version would need costs below its breakeven,
  0.10–0.22 bps per unit traded, or fills better than trade-through, which only quote and order-book data can establish.
<!-- TOP:END -->

## Papers

| Paper | Question | Answer (hypotheses above) |
|---|---|---|
| [1. What the Hasbrouck VAR measures on five-minute bars](paper/paper1.pdf) | Do bar-level order-flow regressions measure information? | Mostly construction identities (H7); the two-step forecast built on the VAR is not identified |
| [2. A multi-timeframe ARX direction model](paper/paper2.pdf) | Is there a forecastable edge, and what is it? | Yes, a one-bar reversal (H1–H3) |
| [3. From gross edge to net profit](paper/paper3.pdf) | Can it be traded? | No: the close-fill edge is bounce, market orders cost more than the edge, resting limit orders are adversely selected (H4–H6) |
| [Passive execution test](docs/PASSIVE_RESULTS.md) ([spec](docs/spec/passive_execution.md)) | Do limit orders rescue it? | No (H6); specification committed before the simulator was written |

The research log, step by step with what each step had to pass: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## Protocol (one for every trading result)

* τ-bars (5m, 15m, 30m, 1h, 2h, 4h) are built inside each session. The first return of a session is
  measured from its open, so no overnight gap enters any series.
* ARX on 5 lags of return and of the flow proxy: rolling 256 sessions, refit every 25.6 sessions.
  Forecasts need only known features; training rows need known targets.
* Decisions are made at the end of every 5-minute bar; each timeframe uses its last finished bar in
  the same session.
* Three fills: the signal bar's close, the next bar's open (primary), and one bar late.
* Positions are closed at every session end, and the closing trade is charged.
* Costs are per stock and per day, from past data only:
  * tick floor (an exact lower bound);
  * Abdi–Ranaldo;
  * Corwin–Schultz.
* Statistics:
  * moving-block bootstrap;
  * deflated Sharpe ratio with 102 counted trials;
  * a fundamental-law check of breadth.
* Look-ahead: truncation-invariance tests at mid-session cuts, plus a positive control with a
  deliberate one-bar leak that the test must catch.

## Layout

```
arxmtf/        data (raw bars -> npz cache), estimation (Hasbrouck VAR, identities), signals (walk-forward ARX),
               costs (tick floor, Abdi-Ranaldo, Corwin-Schultz), backtest (three fills), passive (limit orders), portfolio, stats
scripts/       one script per stage; run_all.py runs them in order; make_papers.py fills the three templates;
               report_readme.py writes the top of this README
tests/         regression against the earlier study (21 checks); look-ahead with positive controls (signals, execution)
paper/         p1..p3_template.tex (with @@placeholders@@), refs.tex, generated paper1..3.tex and PDFs
results/       the JSON/CSV results every paper number comes from (large .npz outputs are not tracked)
docs/          research log, passive-execution specification and results
```

## Reproducing

The raw data are 5-minute bars from Interactive Brokers and are not redistributed. The layout
expected is one folder per ticker containing `*.tsv.gz` files with columns
`timestamp, open, high, low, close, volume`.

```bash
pip install numpy pandas scipy
export ARXMTF_RAW_DATA=/path/to/5_mins
python scripts/run_all.py              # every stage, every test and the three papers (~12 minutes on 8 cores)
latexmk -pdf -cd paper/paper1.tex      # likewise paper2, paper3
```

A full rerun from an empty results folder reproduces every result file and paper byte for byte.

## License

MIT
