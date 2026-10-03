"""Step-1 look-ahead test.

Truncation invariance: run the pipeline on data cut at bar T and on the full data. Every score at
bars <= T (including T itself, the latest bar the truncated run has seen) and every filtered
position must be identical, because nothing computed at time t may depend on data after t.
Cuts are placed at 13:25 (the last 5-minute bar of a 4h, 2h, 1h, 30m and 15m bar, so every
timeframe's bar is complete) and at the session's last bar.
Positive control: inject a feature that uses the next bar (a deliberate leak) and check the test fails.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from arxmtf import config as C, data, signals as S
sys.path.insert(0, os.path.join(C.ROOT, "scripts"))
import run_signals as RS

TICKERS = ["AAPL", "JPM", "XOM"]
CUT_DAYS = [20220315, 20230801, 20241115]


def compare(tic, cut_day, minute):
    d = data.load(tic)
    last = int(np.flatnonzero((d["day"] == cut_day) & (d["minute"] <= minute))[-1])
    full, _, fs = RS.one(tic, save=False)
    trunc, _, ts = RS.one(tic, upto=last, save=False)
    worst = 0.0
    for k in ts:                                   # scores on every 5-minute bar up to the cut
        a, b = fs[k][: last + 1], ts[k]
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            return np.inf
        both = np.isfinite(a)
        worst = max(worst, float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0)
    n = len(trunc["keys"])
    assert np.array_equal(full["keys"][:n], trunc["keys"]), "decision bars differ"
    for k in ("cs", "ar", "tick"):                 # causal cost estimates
        a, b = full[k][:n].astype(float), trunc[k].astype(float)
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            return np.inf
        m = np.isfinite(a)
        worst = max(worst, float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0)
    cols = [f"s_cls_{nm}" for nm, _ in C.TAUS]
    Sf = S.combine(np.column_stack([full[c][:n] for c in cols]).astype(float), np.ones(6))
    St = S.combine(np.column_stack([trunc[c] for c in cols]).astype(float), np.ones(6))
    worst = max(worst, float(np.max(np.abs(S.gated(Sf) - S.gated(St)))))
    return worst


def main():
    fails = 0; n_trunc = 0; n_trunc_ok = 0
    for tic in TICKERS:
        for cd in CUT_DAYS:
            for minute in (805, 955):
                w = compare(tic, cd, minute)
                ok = w < 1e-6
                fails += not ok; n_trunc += 1; n_trunc_ok += ok
                print(f"{'PASS' if ok else 'FAIL'}  truncation {tic} {cd} {minute//60}:{minute%60:02d}: max |diff| = {w:.2e}")
    # positive control: leak the next bar's direction into the proxy
    orig = S.flow_proxy
    S.flow_proxy = lambda O, H, L_, Cl, V, kind="tanh": np.r_[np.sign(Cl[1:] - O[1:]), 0.0]
    try:
        w = compare("AAPL", CUT_DAYS[0], 805)
        caught = w > 1e-6
        fails += not caught
        print(f"{'PASS' if caught else 'FAIL'}  positive control (deliberate leak) detected: max |diff| = {w:.2e}")
    finally:
        S.flow_proxy = orig
    print("FAILURES:", fails)
    json.dump(dict(truncation=n_trunc, truncation_ok=n_trunc_ok, stocks=len(TICKERS), dates=len(CUT_DAYS),
                   control_caught=bool(caught), failures=fails),
              open(os.path.join(C.RESULTS, "test_lookahead.json"), "w"))
    return fails


if __name__ == "__main__":
    sys.exit(main())
