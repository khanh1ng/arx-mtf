# Research log

Requirements R1-R10 are defined in the README. Each step is checked before the next starts.

## Step 1 - reproducible codebase (PASS)

| Check | Requirement | Result |
|---|---|---|
| Estimation numbers reproduce the earlier study | R10 | 14/14 within rounding (b0, b1, a1, R2, IRF, identity, b0 ladder) |
| 5-minute trading replication | R10 | 7/7 (gross 0.1220 bps, Sharpe 6.49, breakeven 0.132, Sharpe@0.25 -5.80, max DD -2.07%; MID Sharpe -7.16) |
| Truncation invariance, 3 stocks x 3 days x 2 cut times (13:25 and session end) | R6 | 18/18, max difference 0 |
| Positive control: a deliberate one-bar leak is caught | R6 | caught (max difference 0.46) |

Issue found and fixed during evaluation: the first version of the look-ahead test cut only at the
session end, where no decision is made, so it could not see a one-bar leak (positive control
failed). Fix: forecasts no longer require the target to be known (as in live use); cuts are placed
mid-session at 13:25, where every timeframe's bar is complete; scores are compared on every bar up to
and including the cut. The change does not alter any decision-bar forecast (replication still exact).

## Step 2 - unified protocol (PASS)

| Check | Requirement | Result |
|---|---|---|
| Every trading number comes from one run with one config (`results/book.npz`, `per_stock.csv`, `metrics.csv`) | R1 | yes: 26 strategies x 3 fills x 5 cost settings x 3 periods |
| Numbers that used the same protocol before are reproduced | R10 | 5m 0.190/0.103/0.030; MTF equal filtered 0.590/0.280/0.125; MTF equal sign 0.318/0.182/0.086; 4h 0.566; MID 5m -0.243 |
| MID identity result holds under the new protocol | R10 | MID 5m accuracy: 0.689 own target, 0.490 close, 0.493 open-to-close; rises to 0.775 at 4h |
| tanh proxy vs plain sign | - | 96% same sign, identical aggregate accuracy (0.5060) and breakeven: checked, not a bug |

Change from before: learned weights are now fitted on the primary (next-open) fill, as decided
before the run. Close-model weights 5m..4h: 0.06, 0.03, 0.02, 0.09, 0.33, 0.44 (normalised).
Exploratory finding, NOT a strategy (chosen after seeing data): the filtered book's breakeven is
1.59 bps in 15:00-15:30 under the next-open fill. Report only as a hypothesis for a future test.

## Step 3 - per-stock costs (PASS, with one flagged estimator)

| Check | Requirement | Result |
|---|---|---|
| Costs per stock, per day, from past data only | R4 | CS, AR from the previous 20 sessions; tick floor from the previous session's close. Added to the truncation test: 18/18 pass |
| Sanity vs large-cap half-spreads (~0.5-2 bps) | R4 | AR median 1.60 bps: pass. CS median 2.31 bps: FAILS (volatility-inflated: TSLA/MRNA/CCL ~6 bps). CS kept only as an upper estimate. Tick floor median 0.47 bps is an exact lower bound |
| All buckets reported, not just the best | R4 | 5 AR quintiles x 4 strategies x all cost models |
| Tight-spread strategy chosen ex ante | R4 | bottom AR quintile, ranked daily on trailing AR |

Decision rule fixed before looking at bucket results: the verdict uses the exact tick floor (a cost
no market order can beat); AR is the central estimate; CS the upper estimate.

Findings:
* Every strategy loses money at the tick floor, full period and hold-out. Closest: 4h alone -2.1%
  (full) / -2.4% (hold-out); learned-weight filtered book -13% / -13%.
* Gross breakeven rises with the AR spread bucket (5m: 0.066 -> 0.197 bps; filtered: 0.135 -> 0.553)
  and per-stock breakeven correlates 0.63 with the stock's AR spread (5m). AR is itself a measure of
  close-vs-midrange bounce, so the edge scales with the bounce: consistent with spread harvesting.
* The AR-tight bucket "breaks even" under AR costs, but ranking and cost use the same noisy
  estimate, so low estimates are selected and then under-charged. Under the tick floor the same
  bucket loses 73% (5m) / 8% (filtered). Not credible as profit.
* Exploratory, NOT pre-specified: by tick-floor quintile, high-priced stocks (floor 0.15 bps) have
  filtered-book breakeven 0.56 bps; 37% of stocks beat their own tick floor. Real spreads of
  high-priced stocks are usually several ticks, so this is a hypothesis, not a result.

## Step 4 - statistics (PASS)

| Check | Requirement | Result |
|---|---|---|
| 95% CIs for gross Sharpe, breakeven, CAGR and max DD at the tick floor, 8 strategies x 2 fills x 2 periods | R2 | done (moving-block bootstrap over days, 20-day blocks, 1,000 draws) |
| CIs contain the point estimates | R2 | yes (e.g. 5m breakeven 0.103 in [0.083, 0.122]; filtered 0.280 in [0.068, 0.504]) |
| Block-length sensitivity (5 / 20 / 60 days) | R2 | intervals move by < 0.04 bps (breakeven) and < 0.4 (Sharpe) |
| Trials counted; deflated Sharpe | R3 | 46 configurations in this run + 39 earlier = 85. Best gross config (5m, widest-spread quintile), Sharpe 6.26: DSR 0.997 (N=85), 0.84 (N=200). 46/46 configurations have negative net Sharpe at the tick floor |
| Fundamental-law paragraph uses measured numbers | R9 | IC 0.011, avg pairwise corr of daily stock P&L 0.025, N_eff 36, predicted portfolio Sharpe 4.70 vs actual 4.76 (daily) |

Implications for the text: (1) the filtered books' breakeven CI is wide ([0.07, 0.50] bps), so
"0.28 vs 0.18" differences between versions are not significant; (2) net CAGR at the tick floor has a
CI entirely below zero for the filtered books (full: [-7.7%, -2.1%]; learned, hold-out: [-7.7%, -1.5%]);
(3) the 4h book alone is indistinguishable from breakeven at the tick floor (CAGR CI [-2.9%, +2.0%]);
(4) the earlier per-stock "Sharpe about 1.4" was a rough bar-level figure; the measured daily
per-stock Sharpe is 0.78 (5m, next-open fill).

## Step 5 - risk and data quality (PASS)

| Check | Requirement | Result |
|---|---|---|
| Beta, net/gross exposure, drawdown duration for the main books | R5 | beta ~0 (|daily corr with market| < 0.04); mean net exposure 0.01-0.03, 99th pct |net| 0.45-0.61; longest gross drawdown 60-367 days |
| Data-quality screen and its effect | R5/R9 | 44 stocks (10%) flagged: >5% of bars with H = L or >1% with zero volume; all high-priced, thinly traded names (NVR, BKNG, AZO, MTD, ...) |

Robustness without the 44 flagged stocks: every book still loses money at the tick floor. Gross
breakeven falls: filtered book 0.280 -> 0.222 (next open), 0.590 -> 0.377 (close fill); learned
filtered 0.309 -> 0.207. Part of the edge comes from stale prints in thin names (unchanged prices
look like reversals). This also explains the exploratory "high-priced stocks" result in Step 3.

## Step 6 - paper (PASS)

| Check | Requirement | Result |
|---|---|---|
| Executive summary on one page with one figure | R8 | page 1: 5 results + equity-curve figure + verdict |
| Main text length | R8 | sections 1-5 on pages 3-10 (8 pages); appendix pages 11-13 |
| Title states the result | R8 | "Why a Sharpe-9 Intraday Signal Loses Money" |
| No hand-typed result numbers | R7/R9 | 169 values filled by `scripts/make_paper.py`; a scan of the template finds only structural numbers (years, 5-minute, S&P 500, citation years, 1.96, bootstrap block lengths) |
| Protocol constants and test outcomes also generated | R9 | from `config.py`, `results/test_*.json` |
| Every trading table from the unified run; old protocol only in Appendix C | R1 | yes |
| CIs, trials, costs, risk, look-ahead all reported | R2-R6 | Tables 9, 6, 14-15, Section 2 |
| Earlier correct findings kept | R10 | MID identity (3.1, App. A), fill bias (3.4) |
| Compiles cleanly | - | 13 pages, no errors, no overfull boxes, no undefined references |

Fixed during evaluation: executive-summary figure had floated to page 2; two figures both numbered 1;
Appendix B tables floated above their heading; protocol constants and test counts were typed by hand.

## Step 7 - reproducibility and final review (PASS)

| Check | Requirement | Result |
|---|---|---|
| Full rerun from an empty results folder (`scripts/run_all.py --skip-cache`, 8m48s) | R7 | `paper/main.tex` and `results/metrics.csv` byte-identical to the previous run; estimation, stats, weights, anatomy identical |
| Nondeterminism found | R7 | `risk.json` differed only because files were read in filesystem order; now sorted |
| Uncited external claim | R9 | "published half-spreads 0.5-2 bps" replaced by a comparison with the paper's own tick-floor range |
| Hand-typed tickers | R9 | "TSLA, MRNA, CCL" was typed from memory and was wrong (top CS stocks are ENPH, CCL, MRNA); now generated |
| All tests | R6/R10 | regression 21/21, truncation 18/18, positive control caught |

Deliverable at this stage: a single paper (later split into the three papers of Round 2).

# Round 2: three papers

Requirements: P1 each paper < 15 pages; P2 abstract, introduction with motivation and contributions,
related work, method, results, discussion, limitations, references; P3 every design choice justified;
P4 every table and figure interpreted; P5 key claims proved or measured; P6 all numbers generated;
P7 each paper stands alone; P8 real references only; P9 compiles cleanly.

## S1 - additional analyses (PASS, one earlier claim corrected)

| Analysis | Result |
|---|---|
| Worked identity example (AAPL 2023-08-01 11:00) | down bar closing 6.64 bps below its midpoint; next midpoint return -0.26 bps = +6.39 forward part - 6.64 known part; next close return +12.26 bps |
| Two-step, unidentified (step A on the same regressors) | rank 11 of 12 in all 17,240 refits; forecasts equal ARX to 1.9e-11 |
| Two-step, identified (step A adds volume and range lags) | delta mean 21.9, positive in 98%, abs(t)>2 in 83% |
| Close-vs-open P&L | A - B = (w_t - w_{t-1}) x (C_t -> O_{t+1}) exactly; measured 0.083 of 0.181 bps/bar (5m); the next print moves -0.11 bps after up bars and +0.15 bps after down bars (87% / 94% of stocks) |
| Sensitivity: lags 1/3/10, windows 128/512, filter quantile 0.5/0.9 | 5m breakeven (next open) 0.089-0.102 bps; filtered 0.141-0.319 bps; every variant loses money at the tick floor (-14% to -98%) |

Correction: the earlier study said the unidentified delta "flips sign like a coin" (positive in 50.9%
of refits). With the pseudo-inverse used here the minimum-norm solution is positive in 91.5% of refits.
The robust facts are the rank deficiency and identical forecasts; the sign claim is dropped.
17 sensitivity configurations are added to the trial count (85 -> 102).

## S2 - Paper 1 "What the Hasbrouck VAR measures on five-minute bars" (PASS)

| Req | Check |
|---|---|
| P1 | 11 pages |
| P2 | abstract; introduction (why measure information, the model, the practical problem, question, 4 findings, why it matters, roadmap); background (theory, the VAR, measuring flow without trades); data and every estimation choice; estimates; three analysis sections; test for bar-level research; limitations; conclusion; references |
| P3 | justified: returns in bps, removing overnight returns (now measured: 7x the 5-minute volatility, not the "ten times" first written), 5 lags, Newey-West, how 431 regressions are summarised, criteria stated before estimation |
| P4 | 6 tables and 2 figures, each followed by a "Reading ..." or test paragraph |
| P5 | Propositions with proofs for the b0 formula and the two-step degeneracy; worked AAPL example; three empirical tests; scatter of fitted vs predicted b1 |
| P6 | 83 generated values; hand-typed "ten times" found and replaced |
| P7 | defines every term; does not rely on the other papers |
| P8 | 10 cited works, all in refs.tex; generator refuses unknown keys |
| P9 | compiles, no errors, no overfull boxes |
Fixed during evaluation: cramped example table, too few decimals for c coefficients.

## S3 - Paper 2 "A multi-timeframe ARX direction model" (PASS)

| Req | Check |
|---|---|
| P1 | 9 pages |
| P2 | abstract; introduction (motivation from the intraday-predictability literature, the idea, why bars, contributions); model; protocol and tests; results (8 subsections); discussion; limitations; conclusion; references |
| P3 | each of the 6 model steps states what and why; window length, refit frequency, fill, measures (why breakeven, not Sharpe) justified |
| P4 | 10 tables and 1 figure, each interpreted |
| P5 | profit decomposition formula; fundamental-law check with measured IC, correlation and breadth; sensitivity over 17 variants |
| P6 | 77 generated values. Found during evaluation: "|s| averages 0.03-0.14" came from a deleted scratch script and was never in the pipeline (it also appeared in the previous single paper); now computed by make_papers.py. Hand-typed "2 and 4 hours", "1- and 2-hour" replaced by generated text; "20,000 bars" corrected to "about 20,000" (256 x 77 = 19,712) |
| P7 | explains the midpoint identity in one paragraph; states the cost result in the discussion |
| P8 | 10 cited works, all real |
| P9 | clean |

## S4 - Paper 3 "From gross edge to net profit" (PASS)

| Req | Check |
|---|---|
| P1 | 9 pages |
| P2 | abstract; introduction (why backtests fail, why reversal is exposed, question, 4 findings); strategy in brief; fill timing; costs; results; what would make it tradable; limitations; conclusion; references |
| P3 | why the next-open fill, why breakeven not Sharpe, why tick floor is the verdict cost, how each estimator works and fails, why buckets by past AR, why block bootstrap, why deflated Sharpe |
| P4 | 7 tables and 1 figure, each interpreted |
| P5 | Proposition (close fill minus next-open fill = position change x jump) with proof and measured bounce; costs from three models; CIs; deflated Sharpe with 102 trials |
| P6 | 83 generated values; time-of-day window and "two- to three-fold" generated |
| P7 | section 2 summarises the strategy so the paper stands alone |
| P8 | 13 cited works, all real |
| P9 | clean after rewording one paragraph (2pt overfull) |

## S5 - cross-check of the three papers (PASS)

* Hand-typed numbers scanned in all three templates; last ones replaced ("431 regressions", "256 / 25.6
  sessions" in the walk-forward figure). Remaining digits are structural (years, timeframe names,
  percentile labels, 1.96, sqrt(2/pi) = 0.80, 45-degree line).
* 10 quantities appear in more than one paper; each is one generated value, so they cannot disagree.
* Each paper cites only keys present in refs.tex (the generator refuses unknown keys).
* Page counts 11 / 9 / 9; all compile with no errors, overfull boxes or undefined references.

## S6 - passive execution test (2026-10-06; procedure PASS, criteria FAIL)

Question: Paper 3 found the gross edge rises with the spread. Can the same positions, traded with limit
orders instead of market orders, keep it?

* Specification written and committed before the simulator (`docs/spec/passive_execution.md`, commit
  `1e5cdb3`): 4 position rules x 2 limit offsets, trade-through fill rule, primary configuration and
  four acceptance criteria fixed in advance.
* Consistency checks: the simulator reproduces the papers' next-open and close-fill P&L on all 431
  stocks (max difference 2e-13 bps); look-ahead identical at 18 truncation points; the positive control
  (an order rule that reads bar t+1) is detected 18 of 18 times.
* Result (`docs/PASSIVE_RESULTS.md`, generated): every configuration is worse than with market orders.
  Fills are adversely selected: about four in five orders fill, the price moves against filled orders
  and in favour of missed ones. A1, A2 and A3 fail; A4 passes.
* Correction: the README had said the edge "belongs to the liquidity provider" and could be used to
  decide where to quote. This test does not support that for a resting order at the last price; both
  statements were replaced.

## S7 - papers brought in line with S6 (2026-10-06)

* Paper 3: the abstract no longer says the edge "is earned by whoever provides liquidity"; Section 6
  gains a paragraph reporting the passive test, with every number generated from `results/passive.json`.
* Paper 2: the sentence saying limit orders "can earn" the bounce now says resting limit orders fail to
  collect it.
* `run_all.py` runs the passive stages before `make_papers.py`. Paper 1 is byte-identical; papers 2 and 3
  keep 9 pages and compile without warnings.
