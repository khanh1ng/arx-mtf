"""Forecasts and scores.

Unified protocol (all trading results):
  * tau-bars are built from 5-minute bars inside each session, starting 09:30;
  * the first return of each session is measured from that session's open, so no overnight gap
    enters any series;
  * a tau-bar is a training row only if the next tau-bar is in the same session;
  * ARX on [1, r_t..r_{t-4}, x_t..x_{t-4}], rolling window of 256 sessions of usable rows,
    refit every 25.6 sessions;
  * score s = tanh(rhat / sigma), sigma = std of that timeframe's returns over the last 20 sessions;
  * on the 5-minute clock a timeframe contributes the score of its latest finished bar, same session.
"""
import numpy as np, pandas as pd
from . import config as C
from .data import lagmat


def aggregate(d, k):
    slot = (d["minute"].astype(np.int64) - C.RTH[0]) // (5 * k)
    gid = d["day"].astype(np.int64) * 1000 + slot
    st = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    en = np.r_[st[1:] - 1, len(gid) - 1]
    return dict(O=d["o"][st], C=d["c"][en], H=np.maximum.reduceat(d["h"], st),
                L=np.minimum.reduceat(d["l"], st), V=np.add.reduceat(d["v"], st),
                day=d["day"][st], end=en, n=len(st))


def session_returns(b, p):
    """bps log returns; first bar of each session measured from the session open."""
    lp = np.log(p)
    first = np.r_[True, b["day"][1:] != b["day"][:-1]]
    r = np.empty(b["n"]); r[1:] = np.diff(lp) * 1e4
    r[first] = (lp[first] - np.log(b["O"][first])) * 1e4
    return r, first


def flow_proxy(O, H, L_, Cl, V, kind="tanh"):
    if kind == "sign":
        return np.sign(Cl - O)
    return np.tanh(np.sign(Cl - O) * np.abs(Cl - O) / (H - L_ + 1e-12) * np.log1p(V))


def walkforward(X, y, train, reest, ridge=0.0, decay=1.0):
    """Rolling OLS / weighted ridge.

    Training rows need a known target; prediction rows need only known features, as in live use.
    Refit points are every `reest` training rows. The model fitted on training rows [p-train, p)
    predicts every row from training row p up to (not including) training row p+reest. All targets
    of those training rows are realised by the time the first predicted row closes.
    Returns (prediction row indices, predictions)."""
    fx = np.all(np.isfinite(X), axis=1)
    tr = np.flatnonzero(fx & np.isfinite(y))
    rows = np.flatnonzero(fx)
    pred = np.full(len(rows), np.nan)
    if tr.size < train + reest:
        return rows, pred
    k = X.shape[1]
    w = decay ** np.arange(train - 1, -1, -1.0) if decay < 1.0 else np.ones(train)
    p = train
    while p < len(tr):
        A = X[tr[p - train:p]]; Aw = A * w[:, None]
        G = Aw.T @ A
        P = np.eye(k) * max(ridge, 1e-10) * np.trace(G) / k; P[0, 0] = 0.0
        beta = np.linalg.solve(G + P, Aw.T @ y[tr[p - train:p]])
        lo = tr[p]; hi = tr[p + reest] if p + reest < len(tr) else X.shape[0]
        sel = (rows >= lo) & (rows < hi)
        pred[sel] = X[rows[sel]] @ beta
        p += reest
    return rows, pred


def tau_model(d, k, price="cls", proxy="tanh", target="same", ridge=0.0, decay=1.0, lags=None, train_days=None):
    """One timeframe. Returns (score on 5m clock, forecast on 5m clock, native accuracy dict)."""
    b = aggregate(d, k)
    p = 0.5 * (b["H"] + b["L"]) if price == "mid" else b["C"]
    r, first = session_returns(b, p)
    rc, _ = session_returns(b, b["C"])
    oc = np.log(b["C"] / b["O"]) * 1e4
    x = flow_proxy(b["O"], b["H"], b["L"], b["C"], b["V"], proxy)
    Lg = lags or C.L; td = train_days or C.TRAIN_DAYS
    X = np.column_stack([np.ones(b["n"]), lagmat(r, range(Lg)), lagmat(x, range(Lg))])
    same = np.r_[~first[1:], False]
    nxt = lambda a: np.where(same, np.r_[a[1:], np.nan], np.nan)
    y = {"same": nxt(r), "oc": nxt(oc)}[target]
    nb = C.NB[k]
    idx, pred = walkforward(X, y, round(td * (nb - 1)), max(1, round(C.REEST_DAYS * (nb - 1))),
                            ridge, decay)
    W = C.SIGMA_DAYS * nb
    sig = pd.Series(r).rolling(W, min_periods=W // 2).std().to_numpy()
    s_bar = np.full(b["n"], np.nan); f_bar = np.full(b["n"], np.nan)
    good = np.isfinite(pred) & np.isfinite(sig[idx]) & (sig[idx] > 0)
    s_bar[idx[good]] = np.tanh(pred[good] / sig[idx[good]])
    f_bar[idx[np.isfinite(pred)]] = pred[np.isfinite(pred)]
    # native accuracy on this timeframe's own next bar
    nat = {}
    for nm, tgt in (("own", nxt(r)), ("cls", nxt(rc)), ("oc", nxt(oc))):
        m = np.isfinite(pred) & np.isfinite(tgt[idx]) & (tgt[idx] != 0) & (pred != 0)
        nat[nm] = (int(np.sum(np.sign(pred[m]) == np.sign(tgt[idx][m]))), int(m.sum()))
    # map to the 5-minute clock: latest finished tau-bar in the same session
    t = np.arange(len(d["c"]))
    j = np.searchsorted(b["end"], t, side="right") - 1
    ok = j >= 0
    ok[ok] &= b["day"][j[ok]] == d["day"][ok]
    s5 = np.full(len(t), np.nan); f5 = np.full(len(t), np.nan)
    s5[ok] = s_bar[j[ok]]; f5[ok] = f_bar[j[ok]]
    return s5, f5, nat


def combine(Smat, weights):
    """Weighted average of available scores; timeframes with no current score are left out."""
    wv = np.asarray(weights, float)
    avail = np.isfinite(Smat)
    num = np.sum(np.where(avail, Smat * wv, 0.0), axis=1)
    den = np.sum(np.where(avail, np.abs(wv), 0.0), axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def gated(S, q=C.GATE_Q, block=C.GATE_BLOCK):
    """sign(S) when |S| >= the q-quantile of |S| over the previous `block` bars, else keep position.
    The first block has no history and stays flat."""
    n = len(S); w = np.zeros(n); prev = 0.0; a = np.abs(S)
    for b0 in range(0, n, block):
        hist = a[max(0, b0 - block):b0]; hist = hist[np.isfinite(hist)]
        thr = np.quantile(hist, q) if (b0 > 0 and hist.size) else np.inf
        seg = range(b0, min(b0 + block, n))
        for t in seg:
            if np.isfinite(S[t]) and a[t] >= thr and S[t] != 0:
                prev = np.sign(S[t])
            w[t] = prev
    return w


def legacy_5m(d, price="cls"):
    """Estimation-protocol 5-minute strategy (replication of the earlier study): overnight return
    dropped, plain-sign proxy, 20,000/2,000-bar walk-forward, target next open-to-close (CLOSE) or
    next midpoint return (MID); earns next close-to-close return."""
    c, o, h, l = d["c"], d["o"], d["h"], d["l"]
    def ret(p):
        r = np.full(len(p), np.nan); r[1:] = np.diff(np.log(p)) * 1e4; r[d["first"]] = np.nan; return r
    rc = ret(c); r = ret(0.5 * (h + l)) if price == "mid" else rc
    x = np.sign(c - o); oc = np.log(c / o) * 1e4
    X = np.column_stack([np.ones(len(c)), lagmat(r, range(C.L)), lagmat(x, range(C.L))])
    same = np.r_[~d["first"][1:], False]
    tgt = r if price == "mid" else oc
    y = np.where(same, np.r_[tgt[1:], np.nan], np.nan)
    idx, pred = walkforward(X, y, C.TRAIN_5M, C.REEST_5M)
    s = np.sign(pred); ycc = np.r_[rc[1:], np.nan][idx]
    m = np.isfinite(pred) & (s != 0) & np.isfinite(ycc)
    return idx[m], s[m], ycc[m]
