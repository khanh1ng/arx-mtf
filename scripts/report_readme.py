"""Write the top of README.md (summary, hypotheses and verdicts, results, mechanism, threats to
validity) between <!-- TOP:START --> and <!-- TOP:END -->. Every number comes from results/: the
paper values (results/paper_values.json, the same strings the papers print), stats.json,
metrics.csv, anatomy.json and passive.json. No number in that block is typed by hand."""
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from arxmtf import config as C

R = C.RESULTS
STRATS = [("cls|5m", "5-minute model"),
          ("cls|eq_sign", "Six timeframes, equal weights"),
          ("cls|eq_filt", "Six timeframes, equal weights, confidence filter"),
          ("cls|lw_filt", "Six timeframes, learned weights, confidence filter")]


def neg(s):
    """Typographic minus for negative numbers only (dates and words keep their hyphens)."""
    return re.sub(r"(?<![\w.])-(?=\d)", "−", s)


def clean(v):
    """Paper value (LaTeX) -> plain text."""
    s = str(v).replace("\\%", "%").replace("{,}", ",").replace("\\$", "\x00").replace("$", "").replace("\x00", "$")
    s = re.sub(r"\\times10\^\{(-?\d+)\}", r"e\1", s)
    return neg(s.replace("--", "–")).strip()


def ci(x, d=2):
    return f"[{x[0]:.{d}f}, {x[1]:.{d}f}]".replace("-", "−")


if __name__ == "__main__":
    V = {k: clean(v) for k, v in json.load(open(os.path.join(R, "paper_values.json"))).items()}
    st = json.load(open(os.path.join(R, "stats.json")))
    an = json.load(open(os.path.join(R, "anatomy.json")))
    pj = json.load(open(os.path.join(R, "passive.json")))
    tp = json.load(open(os.path.join(R, "test_passive.json")))
    tl = json.load(open(os.path.join(R, "test_lookahead.json")))
    m = pd.read_csv(os.path.join(R, "metrics.csv"))
    pm = pd.read_csv(os.path.join(R, "passive_metrics.csv"))
    g = lambda s, per, cost: m[(m.strat == s) & (m.period == per) & (m.fill == "B") & (m.cost == cost)].iloc[0]
    sci = lambda s, per: st["ci"][f"{s}|B|{per}"]["gross_sharpe"]
    xs = [an["coefficients"][f"x_t-{i}"]["pos"] for i in range(5)]
    pr = pm[(pm.cfg == pj["primary"]["cfg"]) & (pm.period == "hold")].iloc[0]
    ph = pm[pm.period == "hold"]
    worse = int(sum(r.tick_bps < g(r.cfg.rsplit("|", 1)[0], "hold", "tick").bps for r in ph.itertuples()))
    assert worse == len(ph), "not every passive configuration is worse than market orders; reword H6"
    be = [g(s, "hold", "0.0").breakeven for s, _ in STRATS]
    p = pj["primary"]

    rows = ["| Strategy | Period | $1M grows to | CAGR | Sharpe [95% CI] | Max drawdown | Months positive | Breakeven (bps) |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s, label in STRATS:
        for per, plabel in (("hold", "Hold-out 2022–2025"), ("full", "Full")):
            r = g(s, per, "0.0")
            note = ", weights learned here" if s == "cls|lw_filt" and per == "full" else ""
            rows.append(f"| {label} | {plabel}, {r.years:.2f} yrs{note} | ${r.end:,.0f} | {r.cagr:+.1%} | "
                        f"{r.sharpe:.2f} {ci(sci(s, per))} | {r.mdd:.1%} | {r.pos_months:.0%} | {r.breakeven:.2f} |"
                        .replace("| -", "| −").replace("[-", "[−").replace(" -", " −"))
    table = "\n".join(rows)

    text = f"""# arx-mtf

**A real but untradable intraday edge: multi-timeframe direction forecasting on {V['nstocks']} S&P 500
stocks.** Three papers and a follow-up execution test on 5-minute bars; evaluation {V['start']} to
{V['end']} ({V['years']} years). Every number below and in the papers is generated from `results/`.

**Summary.** A walk-forward ARX model forecasts the next 5-minute return at six horizons, 5 minutes
to 4 hours. Before costs the forecast has a statistically significant edge: hit rate {V['acc_cls']} and
gross Sharpe {V['f_srB']} (95% CI {ci(sci('cls|5m', 'full'))}) with next-open fills, {g('cls|5m', 'hold', '0.0').sharpe:.2f} {ci(sci('cls|5m', 'hold'))} in the
2022–2025 hold-out. The edge is a one-bar reversal, most of it bid–ask bounce. It breaks even at
{min(be):.2f}–{max(be):.2f} bps per unit traded, while the cheapest possible market order pays {V['tk_med']} bps at the
median stock, and resting limit orders are adversely selected. The contribution is a measured
account of where an intraday edge comes from and what it would cost to capture it.

## Hypotheses and verdicts

| | Hypothesis | Test | Evidence | Verdict |
|---|---|---|---|---|
| H1 | Next-bar returns are forecastable out of sample | Walk-forward ARX, next-open fills; moving-block bootstrap; deflated Sharpe over every configuration run | Hit rate {V['acc_cls']}; gross Sharpe {V['f_srB']} {ci(sci('cls|5m', 'full'))}, hold-out {g('cls|5m', 'hold', '0.0').sharpe:.2f} {ci(sci('cls|5m', 'hold'))}; best of {V['n_all']} configurations: deflated Sharpe probability {V['dsr_all']} | **Supported** |
| H2 | Bar-level order flow adds information | Flow coefficients across {V['n_refits']} refits; proxy replaced by a plain sign | Flow coefficients positive in {min(xs):.0%}–{max(xs):.0%} of refits (no stable sign); only the latest return is stable, negative in {V['beta_r0_neg']} | **Not supported** |
| H3 | Averaging horizons keeps the edge at lower turnover | Equal-weight average of six standardised forecasts against the 5-minute model | Keeps {V['es_keep']} of gross return at {V['es_turnratio']} of turnover; breakeven {V['f_beB']} → {V['es_beB']} bps. Filtered books: hold-out Sharpe CIs {ci(sci('cls|eq_filt', 'hold'))} and {ci(sci('cls|lw_filt', 'hold'))} include zero | **Supported** for equal weights; **not significant** for filtered books |
| H4 | A close-fill backtest measures a tradable edge | Proposition 1: close fill minus next-open fill = position change × jump between the two prints | The jump is bid–ask bounce: {V['dec_share']} of the close-fill edge; Sharpe {V['f_srA']} → {V['f_srB']} | **Rejected** (artifact, proved) |
| H5 | The edge survives market-order costs | Causal per-stock, per-day costs; exact tick floor as lower bound | Breakeven {min(be):.2f}–{max(be):.2f} bps against at least {V['tk_med']} bps; all {V['n_here']} versions lose at the tick floor | **Rejected** |
| H6 | Limit orders collect the spread instead | Passive test with a specification frozen before the run; trade-through fills | {pr.fill_rate:.0%} of orders fill; price moves {neg(f"{pr.move_filled_bps:+.1f}")} bps after fills, {pr.move_unfilled_bps:+.1f} after misses; net Sharpe {neg(f"{p['sharpe_tick']:.1f}")} {ci(p['ci95_sharpe_tick'], 1)}; all {worse} configurations worse than market orders | **Rejected** (adverse selection) |
| H7 | A Hasbrouck VAR on bars measures information in trades | Closed-form identities against the estimates | A regression-free formula matches the impact coefficient (correlation {V['b0_corr']}); one control removes {V['b1_removed']} of the lagged impact | **Rejected** (mostly identity) |

## Results before costs

{table}

Next-open fills, the first price available after each decision; equal-weight book of {V['nstocks']} stocks;
positions closed at every session end. **Net of the smallest possible cost every version loses
money**; the best hold-out version turns $1M into {V['plf_bt_end']}.

## Mechanism: why the edge exists and why it cannot be captured

* **The Sharpe ratio comes from breadth, not accuracy.** The information coefficient is {V['ic']} and a
  single stock's Sharpe ratio is {V['sr_stock']}. With average pairwise correlation {V['rho']} the book holds
  about {V['neff']} independent bets, so the fundamental law predicts {V['sr_pred']}; the realised daily Sharpe
  ratio is {V['sr_act']}.
* **The reversal is the spread.** After a down bar, the jump from the close to the next open reverses the move
  in {V['gap_dn_sh']} of stocks.
  Gross breakeven rises from {V['bkt_f_q1']} to {V['bkt_f_q5']} bps from the tightest to the widest spread quintile
  and correlates {V['corr_be_ar']} with each stock's spread.
* **Filled limit orders are the losing ones.** A buy resting at the last close fills only if the price
  falls through it, which is when the reversal forecast is wrong. Breaking even would need a rebate of
  {p['breakeven_fee_bps'] * -1:.2f} bps per unit traded.

## Threats to validity

| Threat | How it is handled | What remains |
|---|---|---|
| Look-ahead | Truncation-invariance tests on signals ({tl['truncation_ok']} of {tl['truncation']} cuts) and on execution ({len(tp['cases'])} cuts), each with a positive control (signals: {'caught' if tl['control_caught'] else '**missed**'}; execution: {tp['control_detected']} of {tp['control_total']}) | None known |
| Data snooping | Deflated Sharpe over {V['n_all']} configurations; hold-out 2022–2025; timeframe weights learned on 2021 only | Configurations were chosen after earlier results on the same data |
| Fill assumption | Three fill rules; next-open is primary; Proposition 1 quantifies the gap | None |
| Costs | Exact tick-floor lower bound plus Abdi–Ranaldo and Corwin–Schultz, all causal | Bars, not quotes; no market impact, so the negative verdict is conservative |
| Passive fills | Trade-through rule, no queue assumption | Queue position, hidden liquidity and latency not modelled |
| Survivorship | Not handled | Stocks with data through 2025 only |

## Implications

* **For research:** judge a high-turnover strategy by its breakeven cost, not its Sharpe ratio; fill at
  the first print after the signal; and check any bar-level regression for construction identities.
* **For trading:** not tradable as specified. A version would need costs below its breakeven,
  {min(be):.2f}–{max(be):.2f} bps per unit traded, or fills better than trade-through, which only quote and order-book data can establish.
"""
    path = os.path.join(C.ROOT, "README.md")
    s = open(path).read()
    pat = re.compile(r"<!-- TOP:START -->\n.*?<!-- TOP:END -->", re.S)
    assert pat.search(s), "TOP markers missing in README.md"
    open(path, "w").write(pat.sub(lambda mm: "<!-- TOP:START -->\n" + text + "<!-- TOP:END -->", s))
    print("README.md top section written")
