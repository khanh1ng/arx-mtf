"""Full-sample estimation (not trading): Hasbrouck VAR, the MID identity tests, the CLOSE b0 ladder.

Estimation protocol: 5-minute bars, log returns in bps, first return of each session set to missing
(no overnight return), L = 5, Newey-West HAC with bandwidth 10.
"""
import numpy as np
from . import config as C
from .data import load, lagmat


def ols_hac(y, X, maxlags=C.HAC_LAGS, add_const=True):
    """OLS with Newey-West (Bartlett) HAC covariance. Returns beta, se, t, R2, n."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, maxlags + 1):
        G = Xu[L:].T @ Xu[:-L]
        S += (1.0 - L / (maxlags + 1.0)) * (G + G.T)
    se = np.sqrt(np.maximum(np.diag(XtXi @ S @ XtXi), 0.0))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    sst = np.sum((y - y.mean()) ** 2)
    return beta, se, t, (1.0 - u @ u / sst if sst > 0 else np.nan), n


def bar_series(d, drop_overnight=True):
    """Estimation-protocol series for one stock."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    mid = 0.5 * (h + l)
    def ret(p):
        r = np.full(len(p), np.nan)
        r[1:] = np.diff(np.log(p)) * 1e4
        if drop_overnight:
            r[d["first"]] = np.nan
        return r
    return dict(r_mid=ret(mid), r_cls=ret(c), x=np.sign(c - o), oc=np.log(c / o) * 1e4,
                cl=(np.log(c) - np.log(mid)) * 1e4, mid=mid)


def var_one(tic, drop_overnight=True):
    """Hasbrouck VAR, both price definitions, one stock."""
    d = load(tic); s = bar_series(d, drop_overnight)
    L = C.L; x = s["x"]
    out = {"ticker": tic, "drop_overnight": drop_overnight,
           "share_x_pos": float(np.mean(x > 0)), "share_x_zero": float(np.mean(x == 0))}
    for mode, r in (("mid", s["r_mid"]), ("close", s["r_cls"])):
        Rl = lagmat(r, range(1, L + 1))
        b, _, t, r2, n = ols_hac(r, np.column_stack([Rl, lagmat(x, range(0, L + 1))]))
        for i in range(1, L + 1):
            out[f"{mode}_a{i}"], out[f"{mode}_a{i}_t"] = b[i], t[i]
        for j in range(0, L + 1):
            out[f"{mode}_b{j}"], out[f"{mode}_b{j}_t"] = b[L + 1 + j], t[L + 1 + j]
        out[f"{mode}_r2_price"], out[f"{mode}_n"] = r2, n
        b2, _, t2, r22, _ = ols_hac(x, np.column_stack([Rl, lagmat(x, range(1, L + 1))]))
        for i in range(1, L + 1):
            out[f"{mode}_c{i}"], out[f"{mode}_c{i}_t"] = b2[i], t2[i]
        for j in range(1, L + 1):
            out[f"{mode}_d{j}"], out[f"{mode}_d{j}_t"] = b2[L + j], t2[L + j]
        out[f"{mode}_r2_flow"] = r22
    return out


def identity_one(tic):
    """Is MID b1 the close-location identity? Prediction, control regression, CLOSE counterpart."""
    d = load(tic); s = bar_series(d); L = C.L
    x, r, cl, rc = s["x"], s["r_mid"], s["cl"], s["r_cls"]
    Rl = lagmat(r, range(1, L + 1)); X0 = lagmat(x, range(0, L + 1))
    b, _, t, r2, _ = ols_hac(r, np.column_stack([Rl, X0]))
    b2, _, t2, r22, _ = ols_hac(r, np.column_stack([Rl, X0, lagmat(cl, [1])]))
    bc, _, tc, _, _ = ols_hac(rc, np.column_stack([lagmat(rc, range(1, L + 1)), X0]))
    return dict(ticker=tic, b1=b[L + 2], b1_t=t[L + 2], pred_b1=0.5 * (cl[x > 0].mean() - cl[x < 0].mean()),
                b1_ctrl=b2[L + 2], b1_ctrl_t=t2[L + 2], b0=b[L + 1], b0_ctrl=b2[L + 1],
                cls_b1=bc[L + 2], cls_b1_t=tc[L + 2], corr_x_cl=float(np.corrcoef(x, cl)[0, 1]),
                r2=r2, r2_ctrl=r22)


def b0_ladder_one(tic):
    """CLOSE b0 predicted without the regression, on the exact estimation sample."""
    d = load(tic); s = bar_series(d); L = C.L
    rc, x, oc = s["r_cls"], s["x"], s["oc"]
    Xf = np.column_stack([lagmat(rc, range(1, L + 1)), lagmat(x, range(0, L + 1))])
    m = np.isfinite(rc) & np.all(np.isfinite(Xf), axis=1)
    rc, x, oc, Xf = rc[m], x[m], oc[m], Xf[m]
    def reg(y, Z):
        Z = np.column_stack([np.ones(len(y)), Z]); return np.linalg.solve(Z.T @ Z, Z.T @ y)
    sd = oc.std(); ea = np.abs(oc).mean()
    return dict(ticker=tic, sd_oc=sd, Eabs_oc=ea, ratio=ea / sd,
                exkurt=float(((oc - oc.mean()) ** 4).mean() / sd ** 4 - 3), var_x=x.var(),
                p_zero=float(np.mean(x == 0)), corr_gap_x=float(np.corrcoef(rc - oc, x)[0, 1]),
                sd_gap=float((rc - oc).std()), b_oc_uni=float(reg(oc, x)[1]),
                b_rc_uni=float(reg(rc, x)[1]), b_full=float(reg(rc, Xf)[1 + L]))


def irf(a, b, H=12):
    """g_0 = b_0, g_h = b_h + sum_k a_k g_{h-k}; a indexed 1..L (a[0] unused), b indexed 0..L."""
    L = len(b) - 1
    g = np.zeros(H + 1); g[0] = b[0]
    for h in range(1, H + 1):
        g[h] = (b[h] if h <= L else 0.0) + sum(a[k] * g[h - k] for k in range(1, min(h, L) + 1))
    return g
