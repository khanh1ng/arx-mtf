"""Stage P: learn timeframe weights on the learning period, then build every strategy book.
Outputs results/book.npz (per-bar portfolio series), results/per_stock.csv, results/daily_pnl.npz,
results/weights.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from arxmtf import config as C, data, portfolio as PF

PRIMARY_FILL = "B"   # next-open fill (decided before the run)

if __name__ == "__main__":
    tics = data.universe()
    # phase 1: single-timeframe books -> weights from learning-period breakeven (primary fill), floor 0
    only = {f"{p}|{nm}" for p in ("cls", "mid") for nm in PF.TAU_NAMES}
    book, _, _, _ = PF.run(tics, None, only)
    k, n, s = book.series()
    learn = k // 10000 < C.LEARN_END
    weights = {}
    for p in ("cls", "mid"):
        weights[p] = {nm: float(max(s[f"{p}|{nm}|{PRIMARY_FILL}"][learn].mean() /
                                    s[f"{p}|{nm}|turn"][learn].mean(), 0.0)) for nm in PF.TAU_NAMES}
    json.dump(weights, open(os.path.join(C.RESULTS, "weights.json"), "w"), indent=1)
    print("learned weights:", weights)
    # phase 2: everything
    book, per_stock, daily, day_index = PF.run(tics, weights)
    k, n, s = book.series()
    np.savez_compressed(os.path.join(C.RESULTS, "book.npz"), keys=k, n=n, **s)
    pd.DataFrame(per_stock).to_csv(os.path.join(C.RESULTS, "per_stock.csv"), index=False)
    np.savez_compressed(os.path.join(C.RESULTS, "daily_pnl.npz"), days=day_index,
                        **{f"{nm}::{t}": v for nm, dct in daily.items() for t, v in dct.items()})
    print("bars", len(k), "from", k[0], "to", k[-1], "mean names", round(n.mean(), 1), "series", len(s))
