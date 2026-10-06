# arx-mtf

**Why a Sharpe-9 intraday signal loses money.** A 5-minute model on 431 S&P 500 stocks has a real,
steady forecasting edge (hit rate 0.506, gross Sharpe 4.99 with next-open fills, deflated Sharpe
0.978 over 102 trials). The headline Sharpe of 9.01 is a fill assumption: Paper 3 proves 46% of the
gross edge is bid–ask bounce, and every version loses money at the smallest cost a market order can
pay. The papers show how to tell a real edge from an artifact, and a follow-up test, specified before it
ran, shows why limit orders cannot harvest it either. Data: 5-minute bars, December 2019 to January 2025. Every number in the papers
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
* **Passive execution, tested with a frozen specification:** posting limit orders instead of paying
  the spread makes results worse, because fills are adversely selected. Filled orders see the price
  move -4.6 bps against them; missed orders would have gained +25.3 bps
  ([`docs/PASSIVE_RESULTS.md`](docs/PASSIVE_RESULTS.md)).
* **A new result on a standard tool:** on bars, the Hasbrouck VAR's impact estimates are largely
  arithmetic. A formula with no regression matches the estimated impact (correlation 0.9986 across
  stocks), and one control removes 97% of the lagged coefficient.

## Papers

| Paper | Question | Main result |
|---|---|---|
| [1. What the Hasbrouck VAR measures on five-minute bars](paper/paper1.pdf) | Do bar-level order-flow regressions measure information? | Mostly construction identities, with a short test that separates identity from information in any bar-level regression. The two-step forecast built on the VAR is not identified. |
| [2. A multi-timeframe ARX direction model](paper/paper2.pdf) | Is there a forecastable edge, and what is it? | Yes: hit rate 0.506, gross Sharpe 4.99 with next-open fills. It is a one-bar reversal; the midpoint's 0.689 hit rate is an identity. |
| [3. From gross edge to net profit](paper/paper3.pdf) | What does it take to trade it? | The paper proves the gap between a close fill and a next-open fill equals position change times the jump between the two prints (bid–ask bounce, 46% of the gross edge). With market orders paying at least half a $0.01 tick, the 46 versions tested do not break even; the best out-of-sample version turns $1 million into $870,953 over 2022–2025. The gross edge rises with the spread. |

## What the cost results establish

* **Execution is the binding constraint.** The filtered multi-timeframe book can pay at most 0.280
  bps per unit of position change, including adverse selection, to break even. A market order pays
  at least 0.47 bps at the median stock.
* **Providing liquidity does not rescue it.** In a passive execution test specified before it ran
  ([spec](docs/spec/passive_execution.md), [results](docs/PASSIVE_RESULTS.md)), the same positions
  are traded with limit orders at the last close that fill only when the price trades through them.
  82.9% of orders fill, but a fill means the price moved against the forecast. The net Sharpe
  ratio falls to -15.1 (95% interval [-17.1, -13.4]) in the hold-out. Breaking even would
  need a rebate of 3.47 bps per unit traded, about $0.039 per share.
* **What this says about the signal.** A one-bar reversal with a hit rate near 0.506 is real but
  weaker than the information in whether a resting order gets filled. Paper 3 found the gross edge
  rises with the spread; the passive test shows a resting order does not collect that spread on
  this signal. Doing so would need queue position and cancellation speed, which bars cannot show.

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
