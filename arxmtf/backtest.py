"""Positions -> P&L under three fill rules, turnover, and equal-weight portfolio aggregation.

A decision is made at the end of 5-minute bar t (t must have a next bar in the same session).
  close fill  (A): trade at C_t.          pnl = w_t * (C_t -> C_{t+1})
  next open   (B): trade at O_{t+1}.      pnl = w_{t-1} * (C_t -> O_{t+1}) + w_t * (O_{t+1} -> C_{t+1})
  one-bar delay(C): trade at C_{t+1}.     pnl = w_{t-1} * (C_t -> C_{t+1})
Positions are flat overnight: the last position of a session is closed at the session end and the
closing trade is charged. Turnover is |w_t - w_{t-1}| (a long-to-short flip counts 2); cost of a bar
is half-spread x turnover.
"""
import numpy as np
from . import config as C

FILLS = ("A", "B", "C")


def next_returns(d, I):
    lc, lo = np.log(d["c"]), np.log(d["o"])
    return dict(cc=(lc[I + 1] - lc[I]) * 1e4, gap=(lo[I + 1] - lc[I]) * 1e4, oc=(lc[I + 1] - lo[I + 1]) * 1e4)


def contiguity(d, I):
    """True where decision bar I[k] directly follows I[k-1] in the same session."""
    return np.r_[False, (I[1:] == I[:-1] + 1) & (d["day"][I[1:]] == d["day"][I[:-1]])]


def pnl_and_turnover(w, rets, contig):
    w = np.nan_to_num(np.asarray(w, float), nan=0.0)
    wp = np.where(contig, np.r_[0.0, w[:-1]], 0.0)
    close_prev = np.where(contig, 0.0, np.r_[0.0, np.abs(w[:-1])])   # close yesterday's last position
    turn = np.abs(w - wp) + close_prev
    return dict(A=w * rets["cc"], B=wp * rets["gap"] + w * rets["oc"], C=wp * rets["cc"], turn=turn)


def legacy_turnover(s):
    """Turnover convention of the earlier study (no forced overnight close); used only for replication."""
    return np.abs(np.diff(np.r_[0.0, s]))


class Book:
    """Equal-weight cross-sectional book on a fixed calendar of 5-minute keys."""

    def __init__(self, keys):
        self.keys = np.asarray(keys)
        self.n = np.zeros(len(keys))
        self.sums = {}

    def add(self, key, series):
        pos = np.searchsorted(self.keys, key)
        assert np.all(self.keys[pos] == key), "key not in calendar"
        np.add.at(self.n, pos, 1.0)
        for nm, v in series.items():
            if nm not in self.sums:
                self.sums[nm] = np.zeros(len(self.keys))
            np.add.at(self.sums[nm], pos, v)

    def merge(self, other):
        self.n += other.n
        for nm, v in other.sums.items():
            self.sums[nm] = self.sums.get(nm, 0) + v

    def series(self, min_names=50):
        ok = self.n >= min_names
        return self.keys[ok], self.n[ok], {nm: v[ok] / self.n[ok] for nm, v in self.sums.items()}
