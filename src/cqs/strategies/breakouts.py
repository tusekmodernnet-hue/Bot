"""H6 — false breakouts on hourly bars (research/preregistration.md).

A bar's high breaks the highest high of the previous N bars, and within M bars (the breakout bar included)
a bar closes back below that level: the breakout failed. At that close the strategy decides a short, executed
at the next open, with the stop above the highest high since the breakout, the target at the middle of the
N-bar range (checked on closes) and a time exit after 24 bars. Lows mirror it for longs.
Risk per trade is 0.5% of equity at the stop, capped at 1x notional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_PER_TRADE = 0.005
MAX_WEIGHT = 1.0
TIME_EXIT_BARS = 24


def false_breakout(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, n: int, m: int):
    """Weights (NaN = hold) and stop levels for one asset. Every decision at bar i uses bars <= i only."""
    H, L, C = h.to_numpy(), l.to_numpy(), c.to_numpy()
    T = len(C)
    hi_n = h.shift(1).rolling(n, min_periods=n).max().to_numpy()  # previous N bars, current excluded
    lo_n = l.shift(1).rolling(n, min_periods=n).min().to_numpy()
    w = np.full(T, np.nan)
    stop = np.full(T, np.nan)
    pos = 0  # +1 long, -1 short, 0 flat
    stop_lvl = target = np.nan
    exit_at = -1
    up = dn = None  # pending breakouts: (level, range_mid, extreme, deadline)
    for i in range(T):
        if np.isnan(C[i]):
            pos, up, dn = 0, None, None
            continue
        if pos != 0:
            stopped = (pos < 0 and H[i] >= stop_lvl) or (pos > 0 and L[i] <= stop_lvl)
            if stopped:  # the engine fills the stop intrabar; mirror it here
                pos = 0
            elif (pos < 0 and C[i] <= target) or (pos > 0 and C[i] >= target) or i >= exit_at:
                w[i] = 0.0
                pos = 0
            else:
                stop[i] = stop_lvl
            continue
        if np.isnan(hi_n[i]):
            continue
        # new breakouts on this bar
        if H[i] > hi_n[i] and up is None:
            up = [hi_n[i], (hi_n[i] + lo_n[i]) / 2, H[i], i + m - 1]
        if L[i] < lo_n[i] and dn is None:
            dn = [lo_n[i], (hi_n[i] + lo_n[i]) / 2, L[i], i + m - 1]
        if up is not None:
            up[2] = max(up[2], H[i])
            if C[i] < up[0] and i <= up[3]:
                dist = (up[2] - C[i]) / C[i]
                if dist > 0:
                    w[i] = -min(RISK_PER_TRADE / dist, MAX_WEIGHT)
                    pos, stop_lvl, target, exit_at = -1, up[2], up[1], i + TIME_EXIT_BARS
                    stop[i] = stop_lvl
                up = None
            elif i >= up[3]:
                up = None
        if pos == 0 and dn is not None:
            dn[2] = min(dn[2], L[i])
            if C[i] > dn[0] and i <= dn[3]:
                dist = (C[i] - dn[2]) / C[i]
                if dist > 0:
                    w[i] = min(RISK_PER_TRADE / dist, MAX_WEIGHT)
                    pos, stop_lvl, target, exit_at = 1, dn[2], dn[1], i + TIME_EXIT_BARS
                    stop[i] = stop_lvl
                dn = None
            elif i >= dn[3]:
                dn = None
    return pd.Series(w, index=c.index), pd.Series(stop, index=c.index)


def h6_weights(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int, m: int):
    ws, ss = {}, {}
    for a in c.columns:
        ws[a], ss[a] = false_breakout(o[a], h[a], l[a], c[a], n, m)
    return pd.DataFrame(ws), pd.DataFrame(ss)


GRID = [dict(n=n, m=m) for n in (24, 72, 168) for m in (1, 3)]
