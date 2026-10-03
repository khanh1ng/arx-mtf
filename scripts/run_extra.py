"""Stage X: worked identity example, two-step identification check, close-vs-open P&L decomposition
-> results/extra.json"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from multiprocessing import Pool
from arxmtf import config as C, data, signals as S
from arxmtf.data import lagmat


def example():
    d = data.load("AAPL")
    i = int(np.flatnonzero((d["day"] == 20230801) & (d["minute"] == 660))[0])   # 11:00 bar
    o, h, l, c = (float(d[k][i]) for k in "ohlc")
    mid, mid1 = 0.5 * (h + l), 0.5 * (d["h"][i + 1] + d["l"][i + 1])
    return dict(day="2023-08-01", time="11:00", o=o, h=h, l=l, c=c, mid=mid,
                o1=float(d["o"][i + 1]), h1=float(d["h"][i + 1]), l1=float(d["l"][i + 1]), c1=float(d["c"][i + 1]),
                mid1=float(mid1), cl=1e4 * np.log(c / mid), fwd=1e4 * np.log(mid1 / c), rmid=1e4 * np.log(mid1 / mid),
                rcls=1e4 * np.log(d["c"][i + 1] / c), x=float(np.sign(c - o)))


def twostep(tic):
    """Unified protocol, 5-minute close model. Step A: x_{t+1} on Phi; Step B: r_{t+1} on [Phi, xhat]."""
    warnings.simplefilter("ignore")
    d = data.load(tic); b = S.aggregate(d, 1)
    r, first = S.session_returns(b, b["C"])
    x = S.flow_proxy(b["O"], b["H"], b["L"], b["C"], b["V"])
    Phi = np.column_stack([np.ones(b["n"]), lagmat(r, range(C.L)), lagmat(x, range(C.L))])
    extra = np.column_stack([lagmat(np.log1p(b["V"]), range(C.L)), lagmat(np.log(b["H"] / b["L"]) * 1e4, range(C.L))])
    same = np.r_[~first[1:], False]
    yr = np.where(same, np.r_[r[1:], np.nan], np.nan); yx = np.where(same, np.r_[x[1:], np.nan], np.nan)
    ok = np.all(np.isfinite(Phi), 1) & np.all(np.isfinite(extra), 1) & np.isfinite(yr)
    idx = np.flatnonzero(ok); train, reest = round(C.TRAIN_DAYS * 77), round(C.REEST_DAYS * 77)
    out = dict(delta_u=[], t_u=[], delta_i=[], t_i=[], maxdiff=0.0, rank=[], cols=Phi.shape[1] + 1)
    p = train
    while p < len(idx):
        tr = idx[p - train:p]; te = idx[p:min(p + reest, len(idx))]
        P, Q, y, yxx = Phi[tr], np.column_stack([Phi[tr], extra[tr]]), yr[tr], yx[tr]
        # unidentified: step A on Phi itself
        bx = np.linalg.lstsq(P, yxx, rcond=None)[0]; Z = np.column_stack([P, P @ bx])
        G = np.linalg.pinv(Z.T @ Z); bz = G @ Z.T @ y; u = y - Z @ bz
        se = np.sqrt(max(u @ u / (len(y) - Z.shape[1]) * G[-1, -1], 0))
        out["delta_u"].append(bz[-1]); out["t_u"].append(bz[-1] / se if se > 0 else np.nan)
        out["rank"].append(int(np.linalg.matrix_rank(Z)))
        barx = np.linalg.lstsq(P, y, rcond=None)[0]
        Zte = np.column_stack([Phi[te], Phi[te] @ bx])
        out["maxdiff"] = max(out["maxdiff"], float(np.max(np.abs(Zte @ bz - Phi[te] @ barx))))
        # identified: step A has more information than step B
        bq = np.linalg.lstsq(Q, yxx, rcond=None)[0]; Z2 = np.column_stack([P, Q @ bq])
        G2 = np.linalg.inv(Z2.T @ Z2); b2 = G2 @ Z2.T @ y; u2 = y - Z2 @ b2
        se2 = np.sqrt(u2 @ u2 / (len(y) - Z2.shape[1]) * G2[-1, -1])
        out["delta_i"].append(b2[-1]); out["t_i"].append(b2[-1] / se2)
        p += reest
    # close-vs-open decomposition ingredients: jump C_t -> O_{t+1} after up and down bars
    rc = np.log(d["c"][1:] / d["c"][:-1]) * 1e4; gap = np.log(d["o"][1:] / d["c"][:-1]) * 1e4
    samebar = ~d["first"][1:]
    r_t = np.r_[np.nan, rc][:-1]                     # return of bar t
    m = samebar & np.isfinite(r_t)
    g_up, g_dn = gap[m & (r_t > 0)].mean(), gap[m & (r_t < 0)].mean()
    return tic, out, g_up, g_dn


def overnight(tic):
    d = data.load(tic); lc = np.log(d["c"]); lo = np.log(d["o"])
    r = np.diff(lc) * 1e4; f = d["first"][1:]
    return float(np.std(r[f])), float(np.std(r[~f]))


if __name__ == "__main__":
    res = {"example": example()}
    with Pool(C.N_WORKERS) as p:
        on = np.array(p.map(overnight, data.universe()))
    res["overnight"] = dict(sd_on=float(np.median(on[:, 0])), sd_5m=float(np.median(on[:, 1])),
                            ratio=float(np.median(on[:, 0] / on[:, 1])))
    with Pool(C.N_WORKERS) as p:
        rr = p.map(twostep, data.universe())
    du = np.concatenate([np.array(o["delta_u"]) for _, o, _, _ in rr]); tu = np.concatenate([np.array(o["t_u"]) for _, o, _, _ in rr])
    di = np.concatenate([np.array(o["delta_i"]) for _, o, _, _ in rr]); ti = np.concatenate([np.array(o["t_i"]) for _, o, _, _ in rr])
    ranks = np.concatenate([np.array(o["rank"]) for _, o, _, _ in rr])
    res["twostep"] = dict(n_refits=int(len(du)), cols=int(rr[0][1]["cols"]), rank_max=int(ranks.max()),
                          rank_min=int(ranks.min()), maxdiff=float(max(o["maxdiff"] for _, o, _, _ in rr)),
                          u_pos=float(np.mean(du > 0)), u_t2=float(np.nanmean(np.abs(tu) > 2)),
                          u_tmed=float(np.nanmedian(np.abs(tu))),
                          i_mean=float(di.mean()), i_pos=float(np.mean(di > 0)), i_t2=float(np.mean(np.abs(ti) > 2)),
                          i_tmean=float(ti.mean()))
    gu = np.array([g for _, _, g, _ in rr]); gd = np.array([g for _, _, _, g in rr])
    res["gap_after"] = dict(up=float(gu.mean()), down=float(gd.mean()), share_up_neg=float(np.mean(gu < 0)),
                            share_dn_pos=float(np.mean(gd > 0)))
    z = np.load(os.path.join(C.RESULTS, "book.npz"))
    res["decomp"] = {s: dict(A=float(z[f"{s}|A"].mean()), B=float(z[f"{s}|B"].mean()),
                             diff=float((z[f"{s}|A"] - z[f"{s}|B"]).mean()))
                     for s in ("cls|5m", "cls|eq_sign", "cls|eq_filt", "mid|5m")}
    json.dump(res, open(os.path.join(C.RESULTS, "extra.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
