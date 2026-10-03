"""Stage M: metrics for every strategy x fill x cost x period -> results/metrics.csv"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from arxmtf import config as C, stats as ST

if __name__ == "__main__":
    z = np.load(os.path.join(C.RESULTS, "book.npz")); keys = z["keys"]
    strats = sorted({k.rsplit("|", 1)[0] for k in z.files if k.endswith("|turn")})
    periods = {"full": keys > 0, "hold": keys // 10000 >= C.LEARN_END, "learn": keys // 10000 < C.LEARN_END}
    rows = []
    for s in strats:
        u = z[f"{s}|turn"]
        for per, m in periods.items():
            for f in ("A", "B", "C"):
                p = z[f"{s}|{f}"]
                costs = {f"{c}": c for c in C.COST_LEVELS}
                costs["cs"] = z[f"{s}|ccs"]; costs["ar"] = z[f"{s}|car"]; costs["tick"] = z[f"{s}|ctk"]
                for cn, c in costs.items():
                    cc = c if np.isscalar(c) else c[m]
                    rows.append(dict(strat=s, period=per, fill=f, cost=cn, **ST.metrics(p[m], u[m], keys[m], cc)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(C.RESULTS, "metrics.csv"), index=False)
    print(len(df), "rows")
