"""Stage R: market exposure, beta, drawdown duration; data-quality screen and robustness -> results/risk.json"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from arxmtf import config as C, data, portfolio as PF, stats as ST

if __name__ == "__main__":
    z = np.load(os.path.join(C.RESULTS, "book.npz")); keys = z["keys"]; mkt = z["mkt"]
    day = keys // 10000; ud, inv = np.unique(day, return_inverse=True)
    mkt_d = np.bincount(inv, weights=mkt)
    out = {"risk": {}}
    for s in PF.EXPOSURE_STRATS:
        p = z[f"{s}|B"]; pd_ = np.bincount(inv, weights=p)
        Xb = np.column_stack([np.ones(len(p)), mkt]); bb = np.linalg.lstsq(Xb, p, rcond=None)[0]
        Xd = np.column_stack([np.ones(len(pd_)), mkt_d]); bd = np.linalg.lstsq(Xd, pd_, rcond=None)[0]
        m0 = ST.metrics(p, z[f"{s}|turn"], keys, 0.0)
        out["risk"][s] = dict(net_exposure=float(z[f"{s}|net"].mean()), gross_exposure=float(z[f"{s}|gross"].mean()),
                              net_exposure_p99=float(np.percentile(np.abs(z[f"{s}|net"]), 99)),
                              beta_bar=float(bb[1]), beta_day=float(bd[1]),
                              corr_day=float(np.corrcoef(pd_, mkt_d)[0, 1]),
                              dd_days_gross=int(m0["dd_days"]), mdd_gross=float(m0["mdd"]))
    # data quality
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(C.PERTIC, "*.json")))]
    dq = pd.DataFrame([{k: m[k] for k in ("tic", "share_c_eq_h", "share_c_eq_l", "share_v0", "share_hl0",
                                          "n_days", "short_days")} for m in metas])
    flag = dq[(dq.share_hl0 > 0.05) | (dq.share_v0 > 0.01)]
    out["data_quality"] = dict(summary=dq.describe().to_dict(), flagged=flag.tic.tolist(),
                               n_flagged=len(flag))
    # robustness: rebuild the main books without flagged stocks (same weights, same bucket cutpoints)
    tics = [t for t in data.universe() if t not in set(flag.tic)]
    w = json.load(open(os.path.join(C.RESULTS, "weights.json")))
    only = {"cls|5m", "cls|eq_sign", "cls|eq_filt", "cls|lw_filt", "cls|4h", "mid|5m"}
    book, _, _, _ = PF.run(tics, w, only)
    k, n, s2 = book.series()
    out["robust_drop_flagged"] = {nm: dict(
        breakeven_B=float(s2[f"{nm}|B"].mean() / s2[f"{nm}|turn"].mean()),
        breakeven_A=float(s2[f"{nm}|A"].mean() / s2[f"{nm}|turn"].mean()),
        total_tick_B=float(ST.metrics(s2[f"{nm}|B"], s2[f"{nm}|turn"], k, s2[f"{nm}|ctk"])["total"]))
        for nm in only}
    json.dump(out, open(os.path.join(C.RESULTS, "risk.json"), "w"), indent=1, default=float)
    print(json.dumps(out["risk"], indent=0, default=float))
    print("flagged", out["data_quality"]["n_flagged"], out["data_quality"]["flagged"])
    print(json.dumps(out["robust_drop_flagged"], indent=0))
