"""Consistency check 3 for the passive execution test: look-ahead.

Simulating on data truncated at decision bar T must give identical orders (size and limit price) for
every decision bar up to and including T, and identical P&L for every decision whose bar t+1 is
before the cut. Positive control: the "peek" mode, which sends an order only when C_{t+1} is
favourable, cannot be evaluated at T on truncated data (or changes the order there).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from arxmtf import config as C, data, portfolio as PF, passive as PS

TICKERS = ["AAPL", "JPM", "XOM"]
CUT_DAYS = [20220315, 20230801, 20241115]
MINUTES = [805, 950]                # 13:25 and the last decision bar of the session (15:50)


def compare(tic, w, I, contig, cut, mode):
    full = PS.simulate(w, I, contig, data.load(tic), mode=mode)
    upto = int(I[cut])
    tr = PS.simulate(w[: cut + 1], I[: cut + 1], contig[: cut + 1], data.load(tic, upto), mode=mode)
    same_orders = np.array_equal(full["order"][: cut + 1], tr["order"]) and \
        np.array_equal(np.nan_to_num(full["limit"][: cut + 1], nan=-1), np.nan_to_num(tr["limit"], nan=-1))
    same_pnl = np.array_equal(full["pnl"][:cut], tr["pnl"][:cut])
    return bool(same_orders), bool(same_pnl)


def main():
    weights = json.load(open(os.path.join(C.RESULTS, "weights.json")))
    out = {"cases": [], "control": []}
    for tic in TICKERS:
        z = PF._load(tic)
        w = PF.positions(z, weights)["cls|eq_filt"]
        I, contig, keys = z["I"], z["contig"], z["keys"]
        for day in CUT_DAYS:
            for minute in MINUTES:
                hit = np.flatnonzero(keys == day * 10000 + minute)
                if not hit.size:
                    continue
                cut = int(hit[0])
                o, p = compare(tic, w, I, contig, cut, "limit")
                out["cases"].append(dict(tic=tic, key=int(keys[cut]), orders_same=o, pnl_same=p))
                # positive control: force an order at T so the peek rule has something to act on
                w2 = w.copy(); w2[cut] = 2.0     # unreachable target, so an order is always sent at T
                try:
                    o2, _ = compare(tic, w2, I, contig, cut, "peek")
                    detected = not o2
                except IndexError:              # the rule could not be evaluated without bar t+1
                    detected = True
                out["control"].append(dict(tic=tic, key=int(keys[cut]), detected=detected))
    out["pass"] = all(c["orders_same"] and c["pnl_same"] for c in out["cases"])
    out["control_detected"] = sum(c["detected"] for c in out["control"])
    out["control_total"] = len(out["control"])
    json.dump(out, open(os.path.join(C.RESULTS, "test_passive.json"), "w"), indent=1)
    print("look-ahead:", "PASS" if out["pass"] else "FAIL", "| positive control detected in",
          out["control_detected"], "of", out["control_total"])


if __name__ == "__main__":
    main()
