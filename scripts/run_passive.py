"""Stage X: passive (limit-order) execution test. Specification: docs/spec/passive_execution.md.
Outputs results/passive.json and results/passive_metrics.csv."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats as sst
from arxmtf import config as C, data, portfolio as PF, backtest as B, stats as ST, passive as PS

STRATS = ("cls|eq_filt", "cls|5m", "cls|eq_sign", "cls|lw_filt")
KS = (0, 1)
PRIMARY = ("cls|eq_filt", 0)
N_TRIALS = 110
N_BOOT = 1000
BLOCK = 20


def _worker(args):
    tics, keys, weights = args
    book = B.Book(keys)
    per_stock, checks = [], []
    for t in tics:
        z = PF._load(t)
        d = data.load(t)
        I, contig = z["I"], z["contig"]
        pos = PF.positions(z, weights)
        tick, ar = z["tick"].astype(float), z["ar"].astype(float)
        series = {}
        # consistency checks 1 and 2 against the papers' fill formulas, on the raw prices
        rets = B.next_returns(d, I)
        for nm in STRATS:
            ref = B.pnl_and_turnover(pos[nm], rets, contig)
            so = PS.simulate(pos[nm], I, contig, d, mode="open")
            sc = PS.simulate(pos[nm], I, contig, d, mode="close")
            checks.append(dict(tic=t, strat=nm, open=float(np.max(np.abs(so["pnl"] - ref["B"]))),
                               close=float(np.max(np.abs(sc["pnl"] - ref["A"])))))
        price = float(np.mean(d["c"][I]))
        for nm in STRATS:
            for k in KS:
                r = PS.simulate(pos[nm], I, contig, d, k=k)
                tag = f"{nm}|k{k}"
                series[f"{tag}|pnl"] = r["pnl"]
                series[f"{tag}|units"] = r["units"] + r["forced"]
                series[f"{tag}|ctk"] = r["forced"] * np.nan_to_num(tick)
                series[f"{tag}|car"] = r["forced"] * np.nan_to_num(ar)
                sent = r["order"] != 0
                f, u = r["filled"], sent & ~r["filled"]
                per_stock.append(dict(tic=t, cfg=tag, price=price, orders=int(sent.sum()), fills=int(f.sum()),
                                      move_filled=float(np.nansum(r["move"][f])),
                                      move_unfilled=float(np.nansum(r["move"][u])),
                                      pnl=float(r["pnl"].sum()), units=float(r["units"].sum()),
                                      forced=float(r["forced"].sum())))
        book.add(z["keys"], series)
    return book, per_stock, checks


def boot_sharpe(net, keys):
    b = ST.day_blocks(keys); nd = len(b) - 1; rng = np.random.default_rng(1)
    nblk = int(np.ceil(nd / BLOCK)); out = []
    for _ in range(N_BOOT):
        s = rng.integers(0, nd - BLOCK + 1, size=nblk)
        dsel = (s[:, None] + np.arange(BLOCK)[None, :]).ravel()[:nd]
        idx = np.concatenate([np.arange(b[i], b[i + 1]) for i in dsel])
        out.append(ST.sharpe_bar(net[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


if __name__ == "__main__":
    tics = data.universe()
    weights = json.load(open(os.path.join(C.RESULTS, "weights.json")))
    keys = PF.calendar(tics)
    chunks = [tics[i::C.N_WORKERS] for i in range(C.N_WORKERS)]
    with Pool(C.N_WORKERS) as p:
        res = p.map(_worker, [(c, keys, weights) for c in chunks])
    book = res[0][0]
    for b, _, _ in res[1:]:
        book.merge(b)
    ps = pd.DataFrame([r for _, x, _ in res for r in x])
    ck = pd.DataFrame([r for _, _, x in res for r in x])
    k, n, s = book.series()
    periods = {"full": k > 0, "hold": k // 10000 >= C.LEARN_END}
    out = {"checks": dict(open_max_abs_bps=float(ck.open.max()), close_max_abs_bps=float(ck.close.max()),
                          stocks=int(ck.tic.nunique()))}
    rows = []
    for nm in STRATS:
        for kk in KS:
            tag = f"{nm}|k{kk}"
            p, u, ctk, car = (s[f"{tag}|{x}"] for x in ("pnl", "units", "ctk", "car"))
            q = ps[ps.cfg == tag]
            for per, m in periods.items():
                row = dict(cfg=tag, period=per)
                row.update({f"tick_{a}": b for a, b in ST.metrics(p[m], u[m], k[m], ctk[m]).items()})
                row["sharpe_gross"] = ST.sharpe_bar(p[m])
                row["sharpe_ar"] = ST.sharpe_bar(p[m] - car[m])
                row["bps_ar"] = float((p[m] - car[m]).mean())
                row["breakeven_fee_bps"] = float((p[m] - ctk[m]).mean() / u[m].mean())
                row["fill_rate"] = float(q.fills.sum() / q.orders.sum())
                row["move_filled_bps"] = float(q.move_filled.sum() / q.fills.sum())
                row["move_unfilled_bps"] = float(q.move_unfilled.sum() / (q.orders.sum() - q.fills.sum()))
                yrs = pd.Series(p[m] - ctk[m]).groupby(k[m] // 100_000_000).sum()
                row.update({f"net_tick_bps_{int(y)}": float(v) for y, v in yrs.items()})
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS, "passive_metrics.csv"), index=False)
    # primary: bootstrap interval, criteria, deflated Sharpe, breakeven fee in dollars per share
    tag = f"{PRIMARY[0]}|k{PRIMARY[1]}"
    m = periods["hold"]
    net = s[f"{tag}|pnl"][m] - s[f"{tag}|ctk"][m]
    prim = df[(df.cfg == tag) & (df.period == "hold")].iloc[0].to_dict()
    ci = boot_sharpe(net, k[m])
    srs = np.array([ST.sharpe_bar(s[f"{c}|pnl"][m] - s[f"{c}|ctk"][m]) for c in df.cfg.unique()])
    prob, sr0 = ST.deflated_sharpe(ST.sharpe_bar(net), len(net), float(sst.skew(net)),
                                   float(sst.kurtosis(net, fisher=False)), np.resize(srs, N_TRIALS))
    med_price = float(ps[ps.cfg == tag].price.median())
    years_ok = all(prim.get(f"net_tick_bps_{y}", -1) > 0 for y in (2022, 2023, 2024))
    out["primary"] = dict(cfg=tag, period="hold", sharpe_tick=prim["tick_sharpe"], ci95_sharpe_tick=ci,
                          bps_ar=prim["bps_ar"], years=[prim.get(f"net_tick_bps_{y}") for y in (2022, 2023, 2024)],
                          deflated=dict(N=N_TRIALS, prob=prob, expected_max_null=sr0),
                          breakeven_fee_bps=prim["breakeven_fee_bps"],
                          breakeven_fee_usd_per_share_at_median_price=prim["breakeven_fee_bps"] / 1e4 * med_price,
                          median_price=med_price)
    out["criteria"] = dict(A1=bool(ci[0] > 0), A2=bool(prim["bps_ar"] > 0), A3=bool(years_ok),
                           A4_checks_1_2=bool(out["checks"]["open_max_abs_bps"] < 1e-6 and
                                              out["checks"]["close_max_abs_bps"] < 1e-6))
    json.dump(out, open(os.path.join(C.RESULTS, "passive.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    cols = ["cfg", "period", "sharpe_gross", "tick_sharpe", "tick_bps", "bps_ar", "tick_turn", "fill_rate",
            "move_filled_bps", "move_unfilled_bps", "breakeven_fee_bps", "tick_mdd"]
    print(df[cols].round(4).to_string())
