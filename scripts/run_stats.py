"""Stage T: bootstrap confidence intervals, deflated Sharpe, IC / breadth -> results/stats.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats as sst
from arxmtf import config as C, stats as ST, data

MAIN = ["cls|5m", "cls|eq_sign", "cls|eq_size", "cls|eq_filt", "cls|lw_filt", "cls|4h", "mid|5m", "mid|eq_sign"]
N_BOOT = 1000


def boot_stats(p, u, keys, cost_series, block):
    """CI for gross Sharpe, breakeven, and CAGR / max DD net of the tick floor, per-block resampling."""
    b = ST.day_blocks(keys); nd = len(b) - 1; rng = np.random.default_rng(1)
    nblk = int(np.ceil(nd / block)); out = []
    for _ in range(N_BOOT):
        s = rng.integers(0, nd - block + 1, size=nblk)
        dsel = (s[:, None] + np.arange(block)[None, :]).ravel()[:nd]
        idx = np.concatenate([np.arange(b[i], b[i + 1]) for i in dsel])
        pp, uu, cc = p[idx], u[idx], cost_series[idx]
        net = (pp - cc) / 1e4; eq = np.cumprod(1 + net); yrs = nd / 252
        out.append((ST.sharpe_bar(pp), pp.mean() / uu.mean(), eq[-1] ** (1 / yrs) - 1,
                    (eq / np.maximum.accumulate(eq) - 1).min(), ST.sharpe_bar(pp - cc)))
    return np.array(out)


if __name__ == "__main__":
    z = np.load(os.path.join(C.RESULTS, "book.npz")); keys = z["keys"]
    res = {"ci": {}, "block_sensitivity": {}}
    names = ["gross_sharpe", "breakeven", "cagr_tick", "mdd_tick", "sharpe_tick"]
    for per, m in (("full", keys > 0), ("hold", keys // 10000 >= C.LEARN_END)):
        for s in MAIN:
            for f in ("A", "B"):
                p, u, ct = z[f"{s}|{f}"][m], z[f"{s}|turn"][m], z[f"{s}|ctk"][m]
                bs = boot_stats(p, u, keys[m], ct, 20)
                res["ci"][f"{s}|{f}|{per}"] = {n: [float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))]
                                               for i, n in enumerate(names)}
    for s in ("cls|5m", "cls|eq_filt", "cls|lw_filt"):
        p, u, ct = z[f"{s}|B"], z[f"{s}|turn"], z[f"{s}|ctk"]
        res["block_sensitivity"][s] = {}
        for blk in (5, 20, 60):
            bs = boot_stats(p, u, keys, ct, blk)
            res["block_sensitivity"][s][blk] = {n: [float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))]
                                                for i, n in enumerate(names[:2])}
    # deflated Sharpe: all configurations evaluated in this project (gross, next-open fill)
    strats = sorted({k.rsplit("|", 1)[0] for k in z.files if k.endswith("|turn")})
    srs = np.array([ST.sharpe_bar(z[f"{s}|B"]) for s in strats])
    n_trials_here = len(strats)
    n_trials_total = n_trials_here + 9 + 30   # + earlier 5m specification grid (9) and (q, h) frontier (30)
    best = strats[int(np.nanargmax(srs))]; pb = z[f"{best}|B"]
    dsr = {}
    for N in (n_trials_here, n_trials_total, 200):
        sr_trials = np.resize(srs, N)          # variance estimated from the configurations actually run
        prob, sr0 = ST.deflated_sharpe(ST.sharpe_bar(pb), len(pb), float(sst.skew(pb)),
                                       float(sst.kurtosis(pb, fisher=False)), sr_trials)
        dsr[N] = dict(prob=prob, expected_max_null=sr0)
    res["deflated"] = dict(best=best, best_sharpe=float(ST.sharpe_bar(pb)), n_trials_here=n_trials_here,
                           n_trials_total=n_trials_total, by_N=dsr,
                           share_net_tick_negative=float(np.mean([ST.sharpe_bar(z[f"{s}|B"] - z[f"{s}|ctk"]) < 0
                                                                  for s in strats])))
    # IC and breadth for the 5-minute close model and the filtered book
    ps = pd.read_csv(os.path.join(C.RESULTS, "per_stock.csv"))
    sd = {}
    for t in data.universe():
        sd[t] = float(np.load(os.path.join(C.PERTIC, t + ".npz"))["ret_cc"].astype(float).std())
    dp = np.load(os.path.join(C.RESULTS, "daily_pnl.npz"))
    fl = {}
    for s in ("cls|5m", "cls|eq_filt"):
        q = ps[ps.strat == s].copy()
        q["sd"] = q.tic.map(sd); q["active"] = q.n / q.bars
        q["ic"] = (q.pnl_A / q.bars) / (q.sd * np.sqrt(q.active))       # corr(sign, r) on active bars, close fill
        mats = np.array([dp[f"{s}::{t}"] for t in q.tic])                # stocks x days, next-open fill
        mats = mats[:, mats.std(axis=0) > 0] if False else mats
        active = mats[:, np.abs(mats).sum(axis=0) > 0]
        cm = np.corrcoef(active); iu = np.triu_indices_from(cm, 1); rho = float(np.nanmean(cm[iu]))
        sr_i = active.mean(axis=1) / active.std(axis=1) * np.sqrt(252)
        port = active.mean(axis=0); sr_p = port.mean() / port.std() * np.sqrt(252)
        N = active.shape[0]; n_eff = N / (1 + (N - 1) * rho)
        fl[s] = dict(ic_mean=float(q.ic.mean()), ic_median=float(q.ic.median()), sr_stock_mean=float(sr_i.mean()),
                     rho=rho, N=N, n_eff=float(n_eff),
                     sr_port_pred=float(sr_i.mean() * np.sqrt(n_eff)), sr_port_daily=float(sr_p),
                     bets_per_year_per_stock=float((q.n / q.bars).mean() * C.BPY))
    res["fundamental_law"] = fl
    json.dump(res, open(os.path.join(C.RESULTS, "stats.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("deflated", "fundamental_law")}, indent=1, default=float))
    for k in ("cls|5m|B|full", "cls|eq_filt|B|full", "cls|lw_filt|B|hold", "cls|5m|A|full", "cls|4h|B|full"):
        print(k, {n: [round(a, 3), round(b, 3)] for n, (a, b) in res["ci"][k].items()})
    print(json.dumps(res["block_sensitivity"], indent=0))
