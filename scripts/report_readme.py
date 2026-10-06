"""Write the before-cost results table and the cost sensitivity line into README.md, between
<!-- GROSS:START --> and <!-- GROSS:END -->. No number in that block is typed by hand."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from arxmtf import config as C

STRATS = [("cls|5m", "5-minute model"),
          ("cls|eq_sign", "Six timeframes, equal weights"),
          ("cls|eq_filt", "Six timeframes, equal weights, confidence filter"),
          ("cls|lw_filt", "Six timeframes, learned weights, confidence filter")]
PERIODS = [("hold", "Hold-out 2022–2025"), ("full", "Full sample")]

if __name__ == "__main__":
    m = pd.read_csv(os.path.join(C.RESULTS, "metrics.csv"))
    ps = pd.read_csv(os.path.join(C.RESULTS, "per_stock.csv"))
    g = lambda s, per, cost: m[(m.strat == s) & (m.period == per) & (m.fill == "B") & (m.cost == cost)].iloc[0]
    rows = ["| Strategy | Period | $1M grows to | CAGR | Sharpe | Max drawdown | Months positive | Breakeven half-spread (bps) |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s, label in STRATS:
        for per, plabel in PERIODS:
            r = g(s, per, "0.0")
            note = " (weights learned in this period)" if s == "cls|lw_filt" and per == "full" else ""
            rows.append(f"| {label} | {plabel}, {r.years:.2f} yrs{note} | ${r.end:,.0f} | {r.cagr:+.1%} | {r.sharpe:.2f} | {r.mdd:.1%} | "
                        f"{r.pos_months:.0%} | {r.breakeven:.2f} |")
    be = [g(s, "hold", "0.0").breakeven for s, _ in STRATS]
    tick_med = ps[ps.strat == "cls|5m"].mean_tk.median()
    best = max(((g(s, "hold", "tick").end, label) for s, label in STRATS))
    text = "\n".join(rows) + f"""

Before trading costs, with next-open fills (the first price available after each decision); equal-weight
book of 431 stocks, positions closed at every session end.

**But the strategy is very sensitive to transaction costs.** It breaks even at a half-spread of
{min(be):.2f}–{max(be):.2f} bps per unit traded (hold-out). A market order pays at least half a $0.01 tick,
{tick_med:.2f} bps at the median stock, so net of that smallest possible cost every version loses money; the
best hold-out version ({best[1].lower()}) turns $1M into ${best[0]:,.0f}. Limit orders do not help
([`docs/PASSIVE_RESULTS.md`](docs/PASSIVE_RESULTS.md))."""
    p = os.path.join(C.ROOT, "README.md")
    s = open(p).read()
    pat = re.compile(r"(<!-- GROSS:START -->\n).*?(<!-- GROSS:END -->)", re.S)
    assert pat.search(s), "GROSS markers missing in README.md"
    open(p, "w").write(pat.sub(lambda mm: mm.group(1) + text + "\n" + mm.group(2), s))
    print("README.md results table updated")
