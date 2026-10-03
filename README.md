# arx-mtf

**Why a Sharpe-9 intraday signal loses money.** A three-paper study of order flow, forecasting and
trading costs on 5-minute OHLCV bars for 431 S&P 500 stocks (December 2019 – January 2025).
Every number in the papers is generated from the result files.

| Paper | Question | Answer |
|---|---|---|
| [1. What the Hasbrouck VAR measures on five-minute bars](paper/paper1.pdf) | Do bar-level order-flow regressions measure information? | Mostly arithmetic. With close prices the contemporaneous impact is predicted by a formula that uses no regression (correlation 0.9986 across stocks). With range-midpoint prices the lagged impact is a known term displaced by one bar: one control removes 97% of it. The two-step forecast built on the VAR is not identified. |
| [2. A multi-timeframe ARX direction model](paper/paper2.pdf) | Is there a forecastable edge, and what is it? | Yes, small and steady. Hit rate 0.506, gross Sharpe 4.99 with next-open fills. It is a one-bar reversal: the flow proxy adds nothing, and the midpoint's 0.689 hit rate is an identity. Combining six timeframes keeps half the edge at a third of the trading. |
| [3. From gross edge to net profit](paper/paper3.pdf) | Does it survive execution? | No. Filled at the signal bar's close, $1 million grows to $4,091,721 (Sharpe 9.01). The paper proves the gap between a close fill and a next-open fill equals position change times the jump between the two prints, and shows that jump is bid–ask bounce: it removes 46% of the gross edge. All 46 versions lose money at the smallest cost a market order can pay (half a $0.01 tick); the best out-of-sample version turns $1 million into $870,953 over 2022–2025. The edge grows with the spread: it belongs to whoever provides liquidity. |

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
               costs (tick floor, Abdi-Ranaldo, Corwin-Schultz), backtest (three fills), portfolio, stats
scripts/       one script per stage; run_all.py runs them in order; make_papers.py fills the three templates
tests/         regression against the earlier study (21 checks); look-ahead with a positive control
paper/         p1..p3_template.tex (with @@placeholders@@), refs.tex, generated paper1..3.tex and PDFs
results/       the JSON/CSV results every paper number comes from (large .npz outputs are not tracked)
docs/          research log
```

## Reproducing

The raw data are 5-minute bars from Interactive Brokers and are not redistributed. The layout
expected is one folder per ticker containing `*.tsv.gz` files with columns
`timestamp, open, high, low, close, volume`.

```bash
pip install numpy pandas scipy
export ARXMTF_RAW_DATA=/path/to/5_mins
python scripts/run_all.py              # every stage, every test and the three papers (~9 minutes on 8 cores)
latexmk -pdf -cd paper/paper1.tex      # likewise paper2, paper3
```

A full rerun from an empty results folder reproduces every result file and paper byte for byte.

## License

MIT
