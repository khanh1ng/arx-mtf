"""Passive (limit-order) execution simulator. Specification: docs/spec/passive_execution.md.

At the end of decision bar t the order is delta = w_t - a_{t-1}. A buy is posted at L = C_t - k ticks,
a sell at L = C_t + k ticks. It fills in bar t+1 only if the bar trades strictly through L (low < L
for a buy, high > L for a sell), at L, for the full size; otherwise it is cancelled. The position
left at the end of a session is closed by a market order at the last close.

Modes used by the consistency checks: "open" fills every order at O_{t+1} (the papers' next-open
fill); "close" fills every order at C_t (the close fill). "peek" is the positive control for the
look-ahead test: it sends an order only when C_{t+1} is favourable, which uses data after t.
"""
import numpy as np

TICK = 0.01


def simulate(w, I, contig, d, k=0, mode="limit"):
    """w: target positions on the decision bars; I: their raw bar indices; d: raw bars (data.load).

    Returns per-decision arrays: pnl (bps, earned over bar t+1), filled units, forced end-of-session
    units, order (delta sent, 0 if none), limit price, filled flag, and the market move over bar t+1
    in the order's direction (for adverse selection)."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    nbar = len(c)
    n = len(I)
    pnl = np.zeros(n); units = np.zeros(n); forced = np.zeros(n)
    order = np.zeros(n); limit = np.full(n, np.nan); filled = np.zeros(n, bool); move = np.full(n, np.nan)
    w = np.nan_to_num(np.asarray(w, float))
    I = np.asarray(I); contig = np.asarray(contig, bool)
    a = 0.0
    for j in range(n):
        if not contig[j]:
            a = 0.0
        t = int(I[j])
        if t + 1 >= nbar:                      # truncated data: bar t+1 not yet known
            order[j] = w[j] - a
            if order[j] != 0.0:
                if mode == "peek":
                    c[t + 1]                   # the peek rule needs bar t+1: raises IndexError
                limit[j] = c[t] - np.sign(order[j]) * k * TICK
            break
        ct, c1 = c[t], c[t + 1]
        lr = np.log(c1) - np.log(ct)
        p = a * lr * 1e4
        delta = w[j] - a
        if delta != 0.0 and mode == "peek" and not (np.sign(delta) * (c1 - ct) > 0):
            delta = 0.0                        # positive control: uses C_{t+1} to decide at t
        order[j] = delta
        if delta != 0.0:
            s = np.sign(delta)
            if mode == "open":
                L, fill = o[t + 1], True
            elif mode == "close":
                L, fill = ct, True
            else:
                L = ct - s * k * TICK
                fill = bool(l[t + 1] < L) if s > 0 else bool(h[t + 1] > L)
            limit[j] = L
            move[j] = s * lr * 1e4
            if fill:
                p += delta * (np.log(c1) - np.log(L)) * 1e4
                units[j] = abs(delta)
                filled[j] = True
                a += delta
        pnl[j] = p
        if j == n - 1 or not contig[j + 1]:
            forced[j] = abs(a)
            a = 0.0
    return dict(pnl=pnl, units=units, forced=forced, order=order, limit=limit, filled=filled, move=move)
