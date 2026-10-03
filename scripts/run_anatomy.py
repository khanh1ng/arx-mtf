"""Stage N: walk-forward coefficients of the 5-minute close model (unified protocol) and the
correlation of timeframe scores -> results/anatomy.json"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from arxmtf import config as C, data, signals as S
from arxmtf.data import lagmat
TAU = [nm for nm, _ in C.TAUS]


def one(tic):
    warnings.simplefilter("ignore")
    d = data.load(tic); b = S.aggregate(d, 1)
    r, first = S.session_returns(b, b["C"])
    x = S.flow_proxy(b["O"], b["H"], b["L"], b["C"], b["V"])
    X = np.column_stack([np.ones(b["n"]), lagmat(r, range(C.L)), lagmat(x, range(C.L))])
    same = np.r_[~first[1:], False]; y = np.where(same, np.r_[r[1:], np.nan], np.nan)
    tr = np.flatnonzero(np.all(np.isfinite(X), axis=1) & np.isfinite(y))
    train, reest = round(C.TRAIN_DAYS * 77), round(C.REEST_DAYS * 77)
    betas = []; p = train
    while p < len(tr):
        A = X[tr[p - train:p]]; G = A.T @ A
        P = np.eye(A.shape[1]) * 1e-10 * np.trace(G) / A.shape[1]; P[0, 0] = 0
        betas.append(np.linalg.solve(G + P, A.T @ y[tr[p - train:p]])); p += reest
    z = np.load(os.path.join(C.PERTIC, tic + ".npz"))
    out = {}
    for p_ in ("cls", "mid"):
        Sm = np.column_stack([z[f"s_{p_}_{nm}"] for nm in TAU]).astype(float)
        m = np.all(np.isfinite(Sm), axis=1)
        out[p_] = np.corrcoef(Sm[m].T) if m.sum() > 1000 else None
    return np.array(betas), out


if __name__ == "__main__":
    with Pool(C.N_WORKERS) as p:
        res = p.map(one, data.universe())
    B = np.vstack([r[0] for r in res])
    names = ["const"] + [f"r_t-{i}" for i in range(C.L)] + [f"x_t-{i}" for i in range(C.L)]
    anat = {n: dict(mean=float(B[:, j].mean()), pos=float((B[:, j] > 0).mean())) for j, n in enumerate(names)}
    corr = {p_: np.nanmean([r[1][p_] for r in res if r[1][p_] is not None], axis=0).tolist() for p_ in ("cls", "mid")}
    json.dump(dict(coefficients=anat, n_refits=int(len(B)), score_corr=corr),
              open(os.path.join(C.RESULTS, "anatomy.json"), "w"), indent=1)
    print("refits", len(B)); [print(f"{n:8s} {v['mean']: .4f}  pos {v['pos']:.3f}") for n, v in anat.items()]
    print(np.round(np.array(corr["cls"]), 3))
