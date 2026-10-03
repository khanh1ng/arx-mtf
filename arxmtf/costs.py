"""Per-stock, time-varying, causal effective-spread estimates from the bars themselves.

For each stock and day D the half-spread is estimated from 5-minute bars of the previous
SPREAD_LOOKBACK_DAYS sessions only (D itself is excluded), so the cost charged on day D is known
before D opens.
  Corwin-Schultz (2012): two-bar high-low estimator on consecutive bars of the same session,
      negative estimates set to zero, averaged over the window.
  Abdi-Ranaldo (2017): S^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})], eta = (h+l)/2, logs; the
      products are averaged over the window, then S = sqrt(max(avg, 0)).
  Tick floor: a spread cannot be narrower than one tick ($0.01), so a market order pays at least
      $0.005 per share. Half-spread floor (bps) = 0.005 / previous session's last close * 1e4.
      This is an exact lower bound, not an estimate.
Both estimators return full spreads; half-spreads (bps) = full / 2 * 1e4.
"""
import numpy as np
from . import config as C


def _cs_bar(d):
    h, l = np.log(d["h"]), np.log(d["l"])
    n = len(h); out = np.full(n, np.nan)
    beta = (h[1:] - l[1:]) ** 2 + (h[:-1] - l[:-1]) ** 2
    gamma = (np.maximum(h[1:], h[:-1]) - np.minimum(l[1:], l[:-1])) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    out[1:] = np.maximum(s, 0.0)
    out[d["first"]] = np.nan                       # pair would span the overnight
    return out


def _ar_bar(d):
    c, eta = np.log(d["c"]), 0.5 * (np.log(d["h"]) + np.log(d["l"]))
    n = len(c); out = np.full(n, np.nan)
    out[:-1] = 4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    last = np.r_[d["first"][1:], True]             # next bar is another session
    out[last] = np.nan
    return out


def _trailing(d, v, finish):
    """Per-day sums of v -> estimate for each day from the previous LOOKBACK days; mapped to bars."""
    day = d["day"]
    ud, inv = np.unique(day, return_inverse=True)
    ok = np.isfinite(v)
    s = np.bincount(inv[ok], weights=v[ok], minlength=len(ud))
    n = np.bincount(inv[ok], minlength=len(ud)).astype(float)
    cs_, cn_ = np.r_[0, np.cumsum(s)], np.r_[0, np.cumsum(n)]
    k = C.SPREAD_LOOKBACK_DAYS
    i = np.arange(len(ud)); lo = np.maximum(i - k, 0)
    ws, wn = cs_[i] - cs_[lo], cn_[i] - cn_[lo]    # days [i-k, i), excludes day i
    est = np.full(len(ud), np.nan)
    good = (i >= k) & (wn > 0)
    est[good] = finish(ws[good] / wn[good])
    return est[inv], ud, est


def half_spreads(d):
    """Causal per-bar half-spreads (bps) and the per-day table for cross-sectional ranking."""
    cs_bar, cs_ud, cs_day = _trailing(d, _cs_bar(d), lambda m: m / 2 * 1e4)
    ar_bar, _, ar_day = _trailing(d, _ar_bar(d), lambda m: np.sqrt(np.maximum(m, 0)) / 2 * 1e4)
    # tick floor from the previous session's last close
    ud, inv = np.unique(d["day"], return_inverse=True)
    last_close = d["c"][np.r_[np.flatnonzero(np.diff(inv)), len(inv) - 1]]
    prev = np.r_[np.nan, last_close[:-1]]
    tick_day = 0.005 / prev * 1e4
    return dict(cs=cs_bar, ar=ar_bar, tick=tick_day[inv], days=cs_ud, cs_day=cs_day, ar_day=ar_day,
                tick_day=tick_day)
