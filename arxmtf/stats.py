"""Performance metrics, block bootstrap, deflated Sharpe."""
import numpy as np, pandas as pd
from scipy import stats as st
from . import config as C


def metrics(p_bps, turn, keys, cost=0.0, start=1e6):
    """p_bps: per-bar portfolio return in bps; cost: flat half-spread (bps) or per-bar cost series (bps)."""
    net = p_bps - (cost * turn if np.isscalar(cost) else cost)
    r = net / 1e4
    eq = np.cumprod(1.0 + r)
    days = keys // 10000
    yrs = len(np.unique(days)) / 252.0
    peak = np.maximum.accumulate(eq); dd = eq / peak - 1.0
    # longest time under water, in trading days
    under = dd < 0; longest = cur = 0; last_day = None
    for u, dday in zip(under, days):
        if dday != last_day:
            cur = cur + 1 if u else 0
            longest = max(longest, cur); last_day = dday
        elif not u:
            cur = 0
    mon = pd.Series(r).groupby(keys // 1_000_000).apply(lambda x: np.prod(1 + x) - 1)
    sd = r.std(ddof=1)
    return dict(end=start * eq[-1], total=eq[-1] - 1, cagr=eq[-1] ** (1 / yrs) - 1,
                vol=sd * np.sqrt(C.BPY), sharpe=r.mean() / sd * np.sqrt(C.BPY) if sd > 0 else np.nan,
                mdd=dd.min(), dd_days=longest, pos_months=float((mon > 0).mean()), n_months=len(mon),
                years=yrs, bps=float(net.mean()), gross_bps=float(p_bps.mean()), turn=float(turn.mean()),
                breakeven=float(p_bps.mean() / turn.mean()) if turn.mean() > 0 else np.nan)


def day_blocks(keys):
    days = keys // 10000
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    return np.r_[starts, len(keys)]


def block_bootstrap(p_bps, turn, keys, stats_fn, n_boot=1000, block_days=20, seed=0):
    """Moving-block bootstrap over trading days (blocks of `block_days` whole days)."""
    rng = np.random.default_rng(seed)
    b = day_blocks(keys); nd = len(b) - 1
    nblk = int(np.ceil(nd / block_days))
    out = []
    for _ in range(n_boot):
        s = rng.integers(0, nd - block_days + 1, size=nblk)
        dsel = (s[:, None] + np.arange(block_days)[None, :]).ravel()[:nd]
        idx = np.concatenate([np.arange(b[i], b[i + 1]) for i in dsel])
        out.append(stats_fn(p_bps[idx], turn[idx]))
    return np.array(out)


def sharpe_bar(p):
    sd = p.std(ddof=1)
    return p.mean() / sd * np.sqrt(C.BPY) if sd > 0 else np.nan


def deflated_sharpe(sr_ann, n_obs, skew, kurt, sr_trials_ann):
    """Bailey & Lopez de Prado (2014). sr in annual units; converted to per-bar for the test.
    Returns (DSR probability, expected max Sharpe under the null, annual)."""
    f = np.sqrt(C.BPY)
    sr = sr_ann / f
    v = np.var(np.asarray(sr_trials_ann) / f, ddof=1)
    N = len(sr_trials_ann)
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * st.norm.ppf(1 - 1 / N) + g * st.norm.ppf(1 - 1 / (N * np.e)))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(st.norm.cdf(z)), float(sr0 * f)
