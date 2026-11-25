"""Round 5 (research/preregistration.md, "Runda 5"): aggressor flow, liquidation flushes, sweeps with absorption and
the New York opening-range breakout with cheaper execution.

Aggressor flow of a bar: I = (2 * taker_buy_volume - volume) / volume; over several bars, sums in the numerator and
denominator. Every input is used only once its bar has closed; open interest only once its snapshot is available
(`available_time`, 5 minutes after the snapshot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.backtest.limit_sim import Plans
from cqs.strategies import intraday as I

Q = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)


def flow(df: pd.DataFrame, n: int = 1) -> pd.Series:
    """Aggressor flow over the last n bars (the current one included)."""
    net = (2 * df["taker_buy_volume"] - df["volume"]).rolling(n, min_periods=n).sum()
    vol = df["volume"].rolling(n, min_periods=n).sum()
    return net / vol.replace(0, np.nan)


# --------------------------------------------------------------------------------------------------------- R5A
def r5a_z(h1: pd.DataFrame, L: int) -> pd.Series:
    """z-score of the L-hour aggressor flow against its values over the previous 30 days (current bar excluded)."""
    f = flow(h1, L)
    mu = f.rolling(30 * 24, min_periods=30 * 24).mean().shift(1)
    sd = f.rolling(30 * 24, min_periods=30 * 24).std().shift(1)
    return (f - mu) / sd


def r5a_flow(h1: pd.DataFrame, L: int, z_entry: float = 2.0, hold: int = 4):
    """1h bars: |z| > 2 -> trade with the flow for `hold` bars, protective stop 3 x ATR(14)."""
    z = r5a_z(h1, L).to_numpy()
    c, h, l = h1["close"].to_numpy(), h1["high"].to_numpy(), h1["low"].to_numpy()
    entry = np.where(z > z_entry, 1.0, np.where(z < -z_entry, -1.0, 0.0))
    a = I.atr(h1).to_numpy()
    stop = np.where(entry != 0, c - entry * 3 * a, np.nan)
    dl = np.where(entry != 0, np.arange(len(c)) + hold, -1)
    return I.trade_machine(c, h, l, entry, stop, deadline=dl)


# --------------------------------------------------------------------------------------------------------- R5B
def oi_at_close(oi: pd.DataFrame, index: pd.DatetimeIndex, bar: pd.Timedelta) -> pd.Series:
    """Open interest from the last snapshot available at each bar's close (NaN where none is available).
    Snapshots with zero open interest (empty records in the Binance archive: 463 for BTC, 198 for ETH up to
    2025-05) are skipped, as a live system would skip an empty reading."""
    s = oi.set_index("available_time")["sum_open_interest"].sort_index()
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")]
    closes = index + bar
    out = s.reindex(s.index.union(closes)).ffill().reindex(closes)
    first = s.index[0] if len(s) else None
    if first is not None:
        out[closes < first] = np.nan
    out.index = index
    return out


def r5b_features(m15: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    """Per 15m bar, known at its close: return, sigma of the previous 30 days' 15m returns, change of open interest
    between the snapshots available at this and the previous close, aggressor flow."""
    r = m15["close"] / m15["open"] - 1.0
    oi_c = oi_at_close(oi, m15.index, Q)
    return pd.DataFrame({"r": r, "sigma": r.rolling(30 * 96, min_periods=30 * 96).std().shift(1),
                         "d_oi": oi_c / oi_c.shift(1) - 1.0, "flow": flow(m15)}, index=m15.index)


def r5b_flush(m15: pd.DataFrame, h1: pd.DataFrame, oi: pd.DataFrame, k: float, oi_drop: float = 0.01,
              flow_min: float = 0.2, hold: int = 16):
    """Fade a forced-flow bar: |return| > k sigma with open interest falling more than 1% and aggressor flow on the
    side of the move. Exit after `hold` 15m bars (4 hours) or at the protective stop."""
    c, h, l = m15["close"].to_numpy(), m15["high"].to_numpy(), m15["low"].to_numpy()
    x = r5b_features(m15, oi)
    rv, sv, d_oi, f = (x[k].to_numpy() for k in ("r", "sigma", "d_oi", "flow"))
    long_ = (rv < -k * sv) & (d_oi < -oi_drop) & (f < -flow_min)
    short = (rv > k * sv) & (d_oi < -oi_drop) & (f > flow_min)
    entry = np.where(long_, 1.0, np.where(short, -1.0, 0.0))
    sd = I.protective_stop_dist(h1, m15.index)
    stop = np.where(entry != 0, c - entry * sd, np.nan)
    dl = np.where(entry != 0, np.arange(len(c)) + hold, -1)
    return I.trade_machine(c, h, l, entry, stop, deadline=dl)


# --------------------------------------------------------------------------------------------------------- R5C
def sweep_flow(m15: pd.DataFrame, entry: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Aggressor flow over each signal's sweep bars, from the first breach to the reclaim bar (inclusive)."""
    net = np.cumsum(np.r_[0.0, (2 * m15["taker_buy_volume"] - m15["volume"]).to_numpy()])
    vol = np.cumsum(np.r_[0.0, m15["volume"].to_numpy()])
    out = np.full(len(entry), np.nan)
    for i in np.flatnonzero(entry != 0):
        s = starts[i]
        v = vol[i + 1] - vol[s]
        out[i] = (net[i + 1] - net[s]) / v if v > 0 else np.nan
    return out


def r5c_sweep_absorption(m15: pd.DataFrame, theta: float, M: int = 4):
    """R3A sweep (M = 4, no killzones, target = middle of the previous day's range) taken only when the aggressor
    flow over the sweep bars pushed through the level: buyers into a swept high (I > theta) for a short, sellers
    into a swept low (I < -theta) for a long."""
    idx = m15.index
    day = idx.normalize()
    pdh = m15["high"].groupby(day).max().shift(1).reindex(day).to_numpy()
    pdl = m15["low"].groupby(day).min().shift(1).reindex(day).to_numpy()
    c, h, l = m15["close"].to_numpy(), m15["high"].to_numpy(), m15["low"].to_numpy()
    days = day.asi8
    starts = np.full(len(c), -1)
    entry, stop = I._sweeps(c, h, l, pdh, pdl, M, np.ones(len(c), bool), days, starts=starts)
    fl = sweep_flow(m15, entry, starts)
    entry = np.where(((entry < 0) & (fl > theta)) | ((entry > 0) & (fl < -theta)), entry, 0.0)
    tgt = I._targets(entry, stop, c, "mid", ref=(pdh + pdl) / 2)
    entry = np.where(np.isnan(tgt), 0, entry)
    return I.trade_machine(c, h, l, entry, stop, tgt, I.day_last_bar(idx), group=days, per_group="side")


# --------------------------------------------------------------------------------------------------------- R5D
def ny_vol_values(m15: pd.DataFrame, h1: pd.DataFrame, key: np.ndarray) -> pd.DataFrame:
    """Per New York session (rows by session id): position of its first 15m bar, ATR(14, 1h) / price known at
    09:30 (at the close of the bar before the first bar) and the median of the previous 60 sessions' values."""
    a = I.known_at_close(I.atr(h1), H1, m15.index, Q).to_numpy() / m15["close"].to_numpy()
    first = pd.Series(np.arange(len(key)))[key != -1].groupby(key[key != -1]).min()
    vals = pd.Series([a[p - 1] if p > 0 else np.nan for p in first.to_numpy()], index=first.index)
    return pd.DataFrame({"first": first, "value": vals,
                         "median_prev60": vals.rolling(60, min_periods=60).median().shift(1)})


def ny_vol_filter(m15: pd.DataFrame, h1: pd.DataFrame, key: np.ndarray) -> dict:
    """Per New York session: True when its value is at least the median of the previous 60 sessions' values."""
    v = ny_vol_values(m15, h1, key)
    return (v["value"] >= v["median_prev60"]).to_dict()


def r5d_orb_market(m15: pd.DataFrame, h1: pd.DataFrame, minutes: int):
    """R3E on the New York session, only in sessions that pass the volatility filter."""
    c, h, l = m15["close"].to_numpy(), m15["high"].to_numpy(), m15["low"].to_numpy()
    key, end_pos = I.session_keys(m15.index, "ny")
    ok = ny_vol_filter(m15, h1, key)
    entry, stop = _orb_signals(c, h, l, key, end_pos, minutes)
    keep = np.array([k != -1 and ok.get(k, False) for k in key])
    entry = np.where(keep, entry, 0.0)
    return I.trade_machine(c, h, l, entry, stop, deadline=end_pos)


def _orb_signals(c, h, l, key, end_pos, minutes):
    """First close beyond the opening range of each session (as R3E): entry, stop, plus the range edges."""
    T = len(c)
    nbar = minutes // 15
    entry, stop = np.zeros(T), np.full(T, np.nan)
    cur, k, orh, orl, done = None, 0, -np.inf, np.inf, False
    for i in range(T):
        if key[i] == -1:
            continue
        if key[i] != cur:
            cur, k, orh, orl, done = key[i], 0, -np.inf, np.inf, False
        k += 1
        if np.isnan(c[i]):
            done = True
            continue
        if k <= nbar:
            orh, orl = max(orh, h[i]), min(orl, l[i])
            continue
        if done or i >= end_pos[i]:
            continue
        if c[i] > orh:
            entry[i], stop[i], done = 1, orl, True
        elif c[i] < orl:
            entry[i], stop[i], done = -1, orh, True
    return entry, stop


def r5d_orb_limit_plans(m15: pd.DataFrame, h1: pd.DataFrame, idx1: pd.DatetimeIndex, minutes: int) -> Plans:
    """After the first 15m close beyond the opening range (filtered sessions only): a limit order at the broken edge
    (buy at the range high, sell at the range low) valid until 12:00 New York time; stop at the opposite edge; time
    exit at 16:00 New York time."""
    c, h, l = m15["close"].to_numpy(), m15["high"].to_numpy(), m15["low"].to_numpy()
    key, end_pos = I.session_keys(m15.index, "ny")
    ok = ny_vol_filter(m15, h1, key)
    entry, stop = _orb_signals(c, h, l, key, end_pos, minutes)
    rows = []
    for i in np.flatnonzero(entry != 0):
        if not ok.get(key[i], False):
            continue
        t_close = m15.index[i] + Q
        local_day = t_close.tz_convert(I.NY).normalize()
        noon = (local_day + pd.Timedelta(hours=12)).tz_convert("UTC")
        last = (local_day + pd.Timedelta(hours=15, minutes=59)).tz_convert("UTC")
        if t_close >= noon:
            continue
        # the broken edge: for a long, the range high = the opposite of the stop side
        edge = _range_edge(h, l, key, i, minutes, entry[i])
        if (entry[i] > 0 and not c[i] > edge) or (entry[i] < 0 and not c[i] < edge):
            continue  # a limit at the edge would execute at once
        rows.append((t_close, noon - pd.Timedelta(minutes=1), last, entry[i], edge, stop[i]))
    n = len(rows)
    start = idx1.get_indexer([r[0] for r in rows]) if n else np.array([], int)
    end = idx1.get_indexer([r[1] for r in rows]) if n else np.array([], int)
    dl = idx1.get_indexer([r[2] for r in rows]) if n else np.array([], int)
    d = np.array([r[3] for r in rows]) if n else np.array([])
    px = np.array([r[4] for r in rows]) if n else np.array([])
    sl = np.array([r[5] for r in rows]) if n else np.array([])
    nan = np.full(n, np.nan)
    keep = (start >= 0) & (end >= start) & (dl >= 0) if n else np.array([], bool)
    f = lambda x: np.asarray(x)[keep]  # noqa: E731
    return Plans(f(start), f(end), f(np.where(d > 0, px, nan)), f(nan), f(np.where(d > 0, sl, nan)),
                 f(np.where(d < 0, px, nan)), f(nan), f(np.where(d < 0, sl, nan)), f(dl), f(np.full(n, -1)))


def _range_edge(h, l, key, i, minutes, d) -> float:
    """The opening-range edge broken by the signal at bar i (high for a long, low for a short)."""
    nbar = minutes // 15
    first = i
    while first > 0 and key[first - 1] == key[i]:
        first -= 1
    rng = slice(first, first + nbar)
    return float(np.nanmax(h[rng])) if d > 0 else float(np.nanmin(l[rng]))
