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


def false_breakout(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, n: int, m: int,
                   can_enter=None, max_weight: float = MAX_WEIGHT):
    """Weights (NaN = hold) and stop levels for one asset. Every decision at bar i uses bars <= i only.

    `can_enter` (bool per bar) blocks new trades where False; an open trade still runs to its exit.
    """
    w, stop, _ = _false_breakout(h, l, c, n, m, can_enter, max_weight)
    return pd.Series(w, index=c.index), pd.Series(stop, index=c.index)


def _false_breakout(h: pd.Series, l: pd.Series, c: pd.Series, n: int, m: int, can_enter=None,
                    max_weight: float = MAX_WEIGHT):
    """Arrays: weights (NaN = hold), stop levels, and the signed weight of the open trade after each bar's
    decision (0 when flat or exiting), which the portfolio gross cap needs."""
    H, L, C = h.to_numpy(), l.to_numpy(), c.to_numpy()
    T = len(C)
    E = np.ones(T, dtype=bool) if can_enter is None else np.asarray(can_enter, dtype=bool)
    held = np.zeros(T)
    cur_w = 0.0
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
                held[i] = cur_w
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
                if dist > 0 and E[i]:
                    w[i] = cur_w = held[i] = -min(RISK_PER_TRADE / dist, max_weight)
                    pos, stop_lvl, target, exit_at = -1, up[2], up[1], i + TIME_EXIT_BARS
                    stop[i] = stop_lvl
                up = None
            elif i >= up[3]:
                up = None
        if pos == 0 and dn is not None:
            dn[2] = min(dn[2], L[i])
            if C[i] > dn[0] and i <= dn[3]:
                dist = (C[i] - dn[2]) / C[i]
                if dist > 0 and E[i]:
                    w[i] = cur_w = held[i] = min(RISK_PER_TRADE / dist, max_weight)
                    pos, stop_lvl, target, exit_at = 1, dn[2], dn[1], i + TIME_EXIT_BARS
                    stop[i] = stop_lvl
                dn = None
            elif i >= dn[3]:
                dn = None
    return w, stop, held


def cap_gross(w: pd.DataFrame, held: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Scale new entries down so the open trades' gross weight stays within `max_gross`.

    Open trades are never cut; an entry that finds no room gets weight 0 (no trade). Uses only the current bar.
    """
    W, Hd = w.to_numpy(copy=True), held.to_numpy()
    scale = np.ones(W.shape[1])
    for i in range(W.shape[0]):
        entering = ~np.isnan(W[i]) & (W[i] != 0)
        if not entering.any():
            continue
        g_open = np.abs(Hd[i, ~entering] * scale[~entering]).sum()
        g_new = np.abs(W[i, entering]).sum()
        f = min(1.0, max(max_gross - g_open, 0.0) / g_new)
        scale[entering] = f
        W[i, entering] *= f
    return pd.DataFrame(W, index=w.index, columns=w.columns)


def h6_weights(o: pd.DataFrame, h: pd.DataFrame, l: pd.DataFrame, c: pd.DataFrame, n: int, m: int,
               can_enter: pd.DataFrame | None = None, max_weight: dict | None = None,
               max_gross: float | None = None):
    """Weights and stops for every asset. `can_enter`: bool frame aligned to `c` (new trades only where True);
    `max_weight`: per-asset cap (default MAX_WEIGHT); `max_gross`: portfolio cap applied at entry."""
    ws, ss, hs = {}, {}, {}
    for a in c.columns:
        ce = None if can_enter is None else can_enter[a].to_numpy(dtype=bool)
        mw = MAX_WEIGHT if max_weight is None else max_weight.get(a, MAX_WEIGHT)
        ws[a], ss[a], hs[a] = _false_breakout(h[a], l[a], c[a], n, m, ce, mw)
    w = pd.DataFrame(ws, index=c.index)
    if max_gross is not None:
        w = cap_gross(w, pd.DataFrame(hs, index=c.index), max_gross)
    return w, pd.DataFrame(ss, index=c.index)


GRID = [dict(n=n, m=m) for n in (24, 72, 168) for m in (1, 3)]
