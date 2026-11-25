"""Round 4 (research/preregistration.md, "Runda 4"): scalping with limit entries. Order plans for the 1-minute
limit-order simulation (`cqs.backtest.limit_sim`), built from 5m and 1h bars known at the moment of placement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.backtest.limit_sim import Plans
from cqs.strategies.intraday import atr, known_at_close

M5 = pd.Timedelta(minutes=5)
H1 = pd.Timedelta(hours=1)


def _positions(idx1: pd.DatetimeIndex, times: pd.DatetimeIndex) -> np.ndarray:
    """Position of each time in the 1m index (-1 if absent)."""
    p = idx1.get_indexer(times)
    return np.asarray(p)


def _plans(start, end, bp, btp, bsl, sp, stp, ssl, da, dr) -> Plans:
    keep = (start >= 0) & (end >= start) & (~np.isnan(bp) | ~np.isnan(sp))
    f = lambda x: np.asarray(x)[keep]  # noqa: E731
    return Plans(f(start), f(end), f(bp), f(btp), f(bsl), f(sp), f(stp), f(ssl), f(da), f(dr))


def session_vwap(m5: pd.DataFrame) -> pd.Series:
    day = m5.index.normalize()
    tp = (m5["high"] + m5["low"] + m5["close"]) / 3
    pv = (tp * m5["volume"]).groupby(day).cumsum()
    vv = m5["volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def s1_plans(m5: pd.DataFrame, h1: pd.DataFrame, idx1: pd.DatetimeIndex, K: float, trend: bool) -> Plans:
    """VWAP band fade: two-sided limits at VWAP*(1 -/+ K*sigma), valid 5 minutes from each 5m close in
    02:00-22:00 UTC; take profit at that VWAP, stop at the same distance on the other side; time exit 60 min."""
    c = m5["close"]
    vwap = session_vwap(m5)
    d = c / vwap - 1.0
    sig = d.rolling(30 * 288, min_periods=30 * 288).std().shift(1)  # the previous 30 days
    t_close = m5.index + M5
    window = (t_close.hour >= 2) & (t_close.hour < 22)
    bp = (vwap * (1 - K * sig)).where(window)
    sp = (vwap * (1 + K * sig)).where(window)
    bp = bp.where(c > bp)  # a limit that would execute at once is not placed
    sp = sp.where(c < sp)
    if trend:
        ema = h1["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        tr = known_at_close(np.sign(h1["close"] - ema).where(ema.notna()), H1, m5.index, M5)
        bp = bp.where(tr > 0)
        sp = sp.where(tr < 0)
    v = vwap.to_numpy()
    bp, sp = bp.to_numpy(), sp.to_numpy()
    start = _positions(idx1, t_close)
    n = len(start)
    return _plans(start, start + 4, bp, v, bp - (v - bp), sp, v, sp + (sp - v), np.full(n, -1), np.full(n, 60))


def s2_plans(m5: pd.DataFrame, h1: pd.DataFrame, idx1: pd.DatetimeIndex, x: float) -> Plans:
    """At 07:00 UTC: sell limit at Asian high + x*ATR(1h), buy limit at Asian low - x*ATR(1h), valid until 10:00;
    take profit at the range midpoint, stop 1 ATR beyond the entry; exit by 16:00 UTC."""
    idx = m5.index
    day = idx.normalize()
    asia = m5[idx.hour < 6]
    ah = asia["high"].groupby(asia.index.normalize()).max()
    al = asia["low"].groupby(asia.index.normalize()).min()
    a = known_at_close(atr(h1), H1, idx, M5)  # ATR as known at each 5m close
    at = (idx.hour == 6) & (idx.minute == 55)  # the 5m bar that closes at 07:00
    days = day[at]
    A = a[at].to_numpy()
    px = m5["close"][at].to_numpy()
    hi, lo = ah.reindex(days).to_numpy(), al.reindex(days).to_numpy()
    sp, bp = hi + x * A, lo - x * A
    sp = np.where(px < sp, sp, np.nan)
    bp = np.where(px > bp, bp, np.nan)
    mid = (hi + lo) / 2
    start = _positions(idx1, days + pd.Timedelta(hours=7))
    end = _positions(idx1, days + pd.Timedelta(hours=9, minutes=59))
    dl = _positions(idx1, days + pd.Timedelta(hours=15, minutes=59))
    end = np.where(start >= 0, end, -1)
    return _plans(start, end, bp, mid, bp - A, sp, mid, sp + A, dl, np.full(len(days), -1))
