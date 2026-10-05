# arx-mtf

**Why a Sharpe-9 intraday signal loses money.** A 5-minute model on 431 S&P 500 stocks has a real,
steady forecasting edge (hit rate 0.506, gross Sharpe 4.99 with next-open fills, deflated Sharpe
0.978 over 102 trials). The headline Sharpe of 9.01 is a fill assumption: Paper 3 proves 46% of the
gross edge is bid–ask bounce, and every version loses money at the smallest cost a market order can
pay. The papers show how to tell a real edge from an artifact, and that this one belongs to the
liquidity provider. Data: 5-minute bars, December 2019 to January 2025. Every number in the papers
is generated from the result files.

## Highlights

* **A real edge before costs:** hit rate 0.506 and gross Sharpe 4.99 with next-open fills. It
  survives a correction for all 102 configurations tried (deflated Sharpe 0.978).
* **The artifact, measured and proved:** filled at the signal bar's close, $1 million would grow to
  $4,091,721 (Sharpe 9.01). That fill is not achievable; the gap to a next-open fill equals position
  change times the jump between the two prints, and that jump is bid–ask bounce.
* **Net of the smallest possible cost** (half a $0.01 tick), all 46 versions lose money; the best
  out-of-sample version turns $1 million into $870,953 over 2022–2025.
* **Multi-timeframe design works as intended:** combining six timeframes (5 minutes to 4 hours)
  keeps half the edge at a third of the trading; a confidence filter cuts trading by 93%; together
  they raise the edge per trade 1.8 to 2.7 times. Results are robust to the number of lags, the
  training window and the filter level.
* **Where the edge lives:** it grows with the spread (correlation 0.63 with each stock's spread),
  so it is earned by whoever provides liquidity, not by whoever takes it. Passive execution is the
  natural next test; these papers do not run it.
* **A new result on a standard tool:** on bars, the Hasbrouck VAR's impact estimates are largely
  arithmetic. A formula with no regression matches the estimated impact (correlation 0.9986 across
  stocks), and one control removes 97% of the lagged coefficient.

## Papers

| Paper | Question | Main result |
|---|---|---|
| [1. What the Hasbrouck VAR measures on five-minute bars](paper/paper1.pdf) | Do bar-level order-flow regressions measure information? | Mostly construction identities, with a short test that separates identity from information in any bar-level regression. The two-step forecast built on the VAR is not identified. |
| [2. A multi-timeframe ARX direction model](paper/paper2.pdf) | Is there a forecastable edge, and what is it? | Yes: hit rate 0.506, gross Sharpe 4.99 with next-open fills. It is a one-bar reversal; the midpoint's 0.689 hit rate is an identity. |
| [3. From gross edge to net profit](paper/paper3.pdf) | What does it take to trade it? | The paper proves the gap between a close fill and a next-open fill equals position change times the jump between the two prints (bid–ask bounce, 46% of the gross edge). With market orders paying at least half a $0.01 tick, the 46 versions tested do not break even; the best out-of-sample version turns $1 million into $870,953 over 2022–2025. The edge belongs to the liquidity provider. |

**Limits stated in Paper 3:** costs are estimated from bars, not quotes; market impact is not
charged, so the negative verdict is conservative; the sample holds stocks with data through 2025
(survivorship), and the versions were chosen after earlier results on the same data, which the
deflated Sharpe ratio corrects for.

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
