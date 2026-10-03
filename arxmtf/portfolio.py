"""Strategy definitions and the equal-weight book for all trading results (unified protocol)."""
import os
import numpy as np
from multiprocessing import Pool
from . import config as C, signals as S, backtest as B

TAU_NAMES = [nm for nm, _ in C.TAUS]
VARIANT_SIGN = {"cls5_sign": "s_cls5_sign", "mid5_sign": "s_mid5_sign",
                "oc5_tanh": "s_oc5_tanh", "cls5_rd": "s_cls5_rd"}


def positions(z, weights=None, bucket_mask=None):
    """All strategy positions on this stock's decision bars. `weights`: learned per-price weights."""
    pos = {}
    for p in ("cls", "mid"):
        Sm = np.column_stack([z[f"s_{p}_{nm}"] for nm in TAU_NAMES]).astype(float)
        for j, nm in enumerate(TAU_NAMES):
            pos[f"{p}|{nm}"] = np.sign(np.nan_to_num(Sm[:, j]))
        Se = S.combine(Sm, np.ones(len(TAU_NAMES)))
        pos[f"{p}|eq_sign"] = np.sign(np.nan_to_num(Se))
        pos[f"{p}|eq_size"] = np.nan_to_num(Se)
        pos[f"{p}|eq_filt"] = S.gated(Se)
        if weights is not None:
            Sl = S.combine(Sm, [weights[p][nm] for nm in TAU_NAMES])
            pos[f"{p}|lw_sign"] = np.sign(np.nan_to_num(Sl))
            pos[f"{p}|lw_filt"] = S.gated(Sl)
    for nm, col in VARIANT_SIGN.items():
        pos[f"var|{nm}"] = np.sign(np.nan_to_num(z[col].astype(float)))
    return pos


def _load(tic):
    z = np.load(os.path.join(C.PERTIC, tic + ".npz"))
    return {k: z[k] for k in z.files}


def calendar(tics):
    return np.unique(np.concatenate([np.load(os.path.join(C.PERTIC, t + ".npz"))["keys"] for t in tics]))


BUCKET_STRATS = ("cls|5m", "cls|eq_filt", "cls|lw_filt", "cls|eq_sign")
EXPOSURE_STRATS = ("cls|5m", "cls|eq_sign", "cls|eq_size", "cls|eq_filt", "cls|lw_filt", "cls|4h", "mid|5m")
N_BUCKETS = 5


def bucket_cutpoints(tics, day_index):
    """Per-day quintile cutpoints of the causal Abdi-Ranaldo half-spread across stocks."""
    M = np.full((len(tics), len(day_index)), np.nan)
    for i, t in enumerate(tics):
        z = np.load(os.path.join(C.PERTIC, t + ".npz"))
        j = np.searchsorted(day_index, z["sp_days"]); ok = (j < len(day_index))
        ok[ok] &= day_index[j[ok]] == z["sp_days"][ok]
        M[i, j[ok]] = z["ar_day"][ok]
    q = np.nanquantile(M, [0.2, 0.4, 0.6, 0.8], axis=0)   # shape (4, days)
    return q.T


def _worker(args):
    tics, keys, weights, only, day_index, cuts = args
    book = B.Book(keys)
    per_stock = []
    daily = {}
    for t in tics:
        z = _load(t)
        rets = {k: z[f"ret_{k}"].astype(float) for k in ("cc", "gap", "oc")}
        pos = positions(z, weights)
        if only is not None:
            pos = {k: v for k, v in pos.items() if k in only}
        cs, ar, tk = z["cs"].astype(float), z["ar"].astype(float), z["tick"].astype(float)
        series = {"mkt": rets["cc"]}
        di = np.searchsorted(day_index, z["keys"] // 10000)
        if cuts is not None and only is None:
            bkt = np.sum(z["ar"].astype(float)[:, None] > cuts[di], axis=1)      # 0 = tightest quintile
            bkt = np.where(np.isfinite(z["ar"]), bkt, -1)
            for nm in BUCKET_STRATS:
                for b in range(N_BUCKETS):
                    pos[f"{nm}@q{b + 1}"] = np.where(bkt == b, pos[nm], 0.0)
        for nm, w in pos.items():
            r = B.pnl_and_turnover(w, rets, z["contig"])
            for f in B.FILLS:
                series[f"{nm}|{f}"] = r[f]
            series[f"{nm}|turn"] = r["turn"]
            series[f"{nm}|ccs"] = cs * r["turn"]
            series[f"{nm}|car"] = ar * r["turn"]
            series[f"{nm}|ctk"] = tk * r["turn"]
            if nm in EXPOSURE_STRATS:
                series[f"{nm}|net"] = w
                series[f"{nm}|gross"] = np.abs(w)
            hit = (w != 0) & (rets["cc"] != 0)
            right = np.sign(w[hit]) == np.sign(rets["cc"][hit])
            a = np.abs(rets["cc"][hit])
            per_stock.append(dict(tic=t, strat=nm, hits=int(right.sum()), n=int(hit.sum()), bars=len(w),
                                  sum_hit=float(a[right].sum()), sum_miss=float(a[~right].sum()),
                                  pnl_A=float(r["A"].sum()), pnl_B=float(r["B"].sum()),
                                  turn=float(r["turn"].sum()), cost_cs=float((cs * r["turn"]).sum()),
                                  cost_ar=float((ar * r["turn"]).sum()), cost_tk=float((tk * r["turn"]).sum()),
                                  mean_cs=float(np.nanmean(cs)), mean_ar=float(np.nanmean(ar)),
                                  mean_tk=float(np.nanmean(tk))))
            if nm in ("cls|5m", "cls|eq_sign", "cls|eq_filt", "cls|lw_filt"):
                dp = np.zeros(len(day_index)); np.add.at(dp, di, r["B"])
                daily.setdefault(nm, {})[t] = dp
        book.add(z["keys"], series)
    return book, per_stock, daily


def run(tics, weights=None, only=None, cuts_from=None):
    keys = calendar(tics)
    day_index = np.unique(keys // 10000)
    cuts = bucket_cutpoints(cuts_from or tics, day_index) if only is None else None
    chunks = [tics[i::C.N_WORKERS] for i in range(C.N_WORKERS)]
    with Pool(C.N_WORKERS) as p:
        res = p.map(_worker, [(c, keys, weights, only, day_index, cuts) for c in chunks])
    book = res[0][0]
    for b, _, _ in res[1:]:
        book.merge(b)
    per_stock = [r for _, ps, _ in res for r in ps]
    daily = {}
    for _, _, dl in res:
        for nm, dct in dl.items():
            daily.setdefault(nm, {}).update(dct)
    return book, per_stock, daily, day_index
