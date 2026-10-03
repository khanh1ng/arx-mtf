"""Step-1 regression: the rebuilt code must reproduce the numbers of the earlier study."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from arxmtf import config as C, data, backtest as B, stats as ST

EXPECT_EST = {  # value, tolerance
    ("var", "mid_b0", "mean"): (6.1673, 5e-4), ("var", "mid_b1", "mean"): (4.7866, 5e-4),
    ("var", "close_b0", "mean"): (12.2493, 5e-4), ("var", "close_b1", "mean"): (0.0249, 5e-4),
    ("var", "mid_a1", "mean"): (0.1798, 5e-4), ("var", "mid_r2_price", None): (0.348, 5e-4),
    ("var", "close_r2_price", None): (0.420, 5e-4), ("irf", "mid_cum", None): (11.99, 5e-3),
    ("irf", "close_cum", None): (12.07, 5e-3), ("identity", "corr", None): (0.989, 5e-4),
    ("identity", "b1_ctrl", None): (0.128, 5e-4), ("identity", "pred_b1", None): (6.686, 5e-4),
    ("b0_ladder", "pred", None): (12.33, 5e-3), ("b0_ladder", "corr", None): (0.9986, 5e-4),
}
EXPECT_LEGACY = {"cls": dict(gross_bps=(0.1220, 5e-5), sharpe=(6.49, 5e-3), breakeven=(0.132, 5e-4),
                             sharpe_025=(-5.80, 5e-3), mdd=(-0.0207, 5e-4)),
                 "mid": dict(gross_bps=(-0.2407, 5e-4), sharpe=(-7.16, 5e-3))}


def legacy_book(price):
    tics = data.universe(); keys = []; parts = []
    for t in tics:
        z = np.load(os.path.join(C.PERTIC, t + ".npz"))
        k, s, y = z[f"leg_{price}_keys"], z[f"leg_{price}_s"].astype(float), z[f"leg_{price}_y"].astype(float)
        parts.append((k, s * y, B.legacy_turnover(s))); keys.append(k)
    book = B.Book(np.unique(np.concatenate(keys)))
    for k, p, u in parts:
        book.add(k, {"p": p, "u": u})
    return book.series()


def main():
    est = json.load(open(os.path.join(C.RESULTS, "estimation.json")))
    fails = 0
    for (a, b, c), (v, tol) in EXPECT_EST.items():
        got = est[a][b][c] if c else est[a][b]
        ok = abs(got - v) <= tol + 1e-12 or abs(round(got, len(str(v).split(".")[1])) - v) < 1e-12
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {a}.{b}{'.' + c if c else ''}: got {got:.5f} expected {v}")
    for price, exp in EXPECT_LEGACY.items():
        k, n, s = legacy_book(price)
        m0 = ST.metrics(s["p"], s["u"], k, 0.0); m1 = ST.metrics(s["p"], s["u"], k, 0.25)
        got = dict(gross_bps=m0["gross_bps"], sharpe=m0["sharpe"], breakeven=m0["breakeven"],
                   sharpe_025=m1["sharpe"], mdd=m0["mdd"])
        for nm, (v, tol) in exp.items():
            ok = abs(got[nm] - v) <= tol
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  legacy {price} {nm}: got {got[nm]:.5f} expected {v}")
    print("FAILURES:", fails)
    n_checks = len(EXPECT_EST) + sum(len(v) for v in EXPECT_LEGACY.values())
    json.dump(dict(checks=n_checks, failures=fails), open(os.path.join(C.RESULTS, "test_regression.json"), "w"))
    return fails


if __name__ == "__main__":
    sys.exit(main())
