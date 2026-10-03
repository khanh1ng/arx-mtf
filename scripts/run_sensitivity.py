"""Stage V: robustness of the close model to lags, training window and filter quantile.
Reported as robustness, not used to choose a configuration. Evaluation window: 2021-06-01 onward,
the latest start among all variants (the 512-session window needs more history)."""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from arxmtf import config as C, data, signals as S, backtest as B, stats as ST

VARIANTS = {"base": {}, "L1": dict(lags=1), "L3": dict(lags=3), "L10": dict(lags=10),
            "W128": dict(train_days=128), "W512": dict(train_days=512)}
QS = {"q050": 0.5, "q075": 0.75, "q090": 0.9}
START = 20210601


def one(args):
    tic, vname = args
    warnings.simplefilter("ignore")
    d = data.load(tic); n = len(d["c"])
    Sm = np.column_stack([S.tau_model(d, k, "cls", **VARIANTS[vname])[0] for _, k in C.TAUS])
    dec = np.r_[~d["first"][1:], False]
    I = np.flatnonzero(dec & (d["day"] >= START) & np.all(np.isfinite(Sm) | (d["day"] >= 0)[:, None], axis=1))
    I = I[np.isfinite(Sm[I, 0])]
    rets = B.next_returns(d, I); contig = B.contiguity(d, I)
    from arxmtf import costs
    tk = costs.half_spreads(d)["tick"][I]
    Se = S.combine(Sm[I], np.ones(6))
    pos = {"5m": np.sign(np.nan_to_num(Sm[I, 0])), "eq_sign": np.sign(np.nan_to_num(Se))}
    for qn, q in QS.items():
        if vname == "base" or qn == "q075":
            pos[f"eq_filt_{qn}"] = S.gated(Se, q)
    out = {}
    for nm, w in pos.items():
        r = B.pnl_and_turnover(w, rets, contig)
        out[nm] = dict(A=r["A"], B=r["B"], turn=r["turn"], ctk=tk * r["turn"])
    return vname, d["key"][I], out


if __name__ == "__main__":
    tics = data.universe()
    jobs = [(t, v) for v in VARIANTS for t in tics]
    with Pool(C.N_WORKERS) as p:
        res = p.map(one, jobs, chunksize=4)
    books = {}
    for vname, keys, out in res:
        for nm, ser in out.items():
            books.setdefault((vname, nm), []).append((keys, ser))
    table = {}
    for (vname, nm), parts in books.items():
        cal = np.unique(np.concatenate([k for k, _ in parts]))
        bk = B.Book(cal)
        for k, ser in parts:
            bk.add(k, ser)
        k, n, s = bk.series()
        m0 = ST.metrics(s["B"], s["turn"], k, 0.0); mt = ST.metrics(s["B"], s["turn"], k, s["ctk"])
        table[f"{vname}|{nm}"] = dict(beA=float(s["A"].mean() / s["turn"].mean()), beB=m0["breakeven"],
                                      srB=m0["sharpe"], turn=m0["turn"], tot_tick=mt["total"],
                                      start=int(k[0] // 10000), end=int(k[-1] // 10000))
    json.dump(table, open(os.path.join(C.RESULTS, "sensitivity.json"), "w"), indent=1)
    for key in sorted(table):
        v = table[key]; print(f"{key:24s} beA {v['beA']:.3f} beB {v['beB']:.3f} srB {v['srB']:.2f} turn {v['turn']:.3f} tick {v['tot_tick']:+.3f} {v['start']}")
