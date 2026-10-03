"""Stage S: per-stock scores, forecasts, native accuracies, spreads -> results/pertic/<tic>.npz"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import numpy as np
from multiprocessing import Pool
from arxmtf import config as C, data, signals as S, costs, backtest as B

VARIANTS = {   # 5-minute model variants (unified protocol); name -> tau_model kwargs
    "cls5_sign":  dict(k=1, price="cls", proxy="sign"),
    "mid5_sign":  dict(k=1, price="mid", proxy="sign"),
    "oc5_tanh":   dict(k=1, price="cls", proxy="tanh", target="oc"),
    "cls5_rd":    dict(k=1, price="cls", proxy="tanh", ridge=1e-3, decay=0.9998),
}


def one(tic, upto=None, save=True):
    warnings.simplefilter("ignore")
    d = data.load(tic, upto)
    n = len(d["c"])
    scores, fc, nat = {}, {}, {}
    for price in ("cls", "mid"):
        for nm, k in C.TAUS:
            s5, f5, na = S.tau_model(d, k, price)
            scores[f"{price}_{nm}"], fc[f"{price}_{nm}"], nat[f"{price}_{nm}"] = s5, f5, na
    for vn, kw in VARIANTS.items():
        k = kw.pop("k"); s5, f5, na = S.tau_model(d, k, **kw); kw["k"] = k
        scores[vn], fc[vn], nat[vn] = s5, f5, na
    # common evaluation start: every score series has started
    t0 = max(int(np.argmax(np.isfinite(v))) if np.isfinite(v).any() else n for v in scores.values())
    dec = np.r_[~d["first"][1:], False]
    I = np.flatnonzero(dec & (np.arange(n) >= t0))
    rets = B.next_returns(d, I)
    mid = 0.5 * (d["h"] + d["l"])
    dmid = (np.log(mid[I + 1]) - np.log(mid[I])) * 1e4
    # 5-minute accuracy of each 5m forecast against three next-bar returns, on the decision bars
    acc5 = {}
    for nm in ["cls_5m", "mid_5m"] + list(VARIANTS):
        f = fc[nm][I]
        for tn, y in (("mid", dmid), ("cls", rets["cc"]), ("oc", rets["oc"])):
            m = np.isfinite(f) & (f != 0) & (y != 0)
            acc5[f"{nm}|{tn}"] = (int(np.sum(np.sign(f[m]) == np.sign(y[m]))), int(m.sum()))
    sp = costs.half_spreads(d)
    li, ls, ly = S.legacy_5m(d, "cls"); mi, ms, my = S.legacy_5m(d, "mid")
    out = dict(keys=d["key"][I], I=I, contig=B.contiguity(d, I),
               **{f"ret_{k}": v.astype(np.float32) for k, v in rets.items()},
               dmid=dmid.astype(np.float32),
               **{f"s_{k}": v[I].astype(np.float32) for k, v in scores.items()},
               cs=sp["cs"][I].astype(np.float32), ar=sp["ar"][I].astype(np.float32), tick=sp["tick"][I].astype(np.float32),
               sp_days=sp["days"], cs_day=sp["cs_day"].astype(np.float32), ar_day=sp["ar_day"].astype(np.float32), tick_day=sp["tick_day"].astype(np.float32),
               leg_cls_keys=d["key"][li], leg_cls_s=ls.astype(np.float32), leg_cls_y=ly.astype(np.float32),
               leg_mid_keys=d["key"][mi], leg_mid_s=ms.astype(np.float32), leg_mid_y=my.astype(np.float32))
    meta = dict(tic=tic, t0=t0, n_dec=len(I), nat=nat, acc5=acc5,
                share_c_eq_h=float(np.mean(d["c"] == d["h"])), share_c_eq_l=float(np.mean(d["c"] == d["l"])),
                share_v0=float(np.mean(d["v"] == 0)), share_hl0=float(np.mean(d["h"] == d["l"])),
                n_days=int(len(np.unique(d["day"]))), short_days=int(np.sum(np.bincount(
                    np.unique(d["day"], return_inverse=True)[1]) < C.BARS_PER_DAY)))
    if save:
        np.savez_compressed(os.path.join(C.PERTIC, tic + ".npz"), **out)
        json.dump(meta, open(os.path.join(C.PERTIC, tic + ".json"), "w"))
    return out, meta, scores


def _run(tic):
    one(tic); return tic


if __name__ == "__main__":
    os.makedirs(C.PERTIC, exist_ok=True)
    tics = data.universe()
    with Pool(C.N_WORKERS) as p:
        done = p.map(_run, tics, chunksize=2)
    print("done", len(done))
