"""Round 3 (research/preregistration.md, "Runda 3"): SMC/ICT, day-trading and swing families on BTC and ETH.

Every family builds, per contract, arrays of entry decisions taken at a bar's close (direction, stop, target,
deadline) from data up to that close only. `trade_machine` turns them into engine weights (NaN = hold) and stop
levels with risk-based sizing. Strategies on 1h or 4h bars are mapped onto 15m bars so that every variant of a
family runs on the same bars: a decision at the close of an hour is placed on the last 15m bar of that hour.

Calendar facts (which bar is the last of a UTC day or of the New York session) are known in advance and are
not market information; prices, volumes and levels are only used once their bar has closed.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

RISK_PER_TRADE = 0.005
MAX_WEIGHT = 1.0
NY = ZoneInfo("America/New_York")
KILLZONE_HOURS = (7, 8, 9, 12, 13, 14)  # bars opening 07:00-09:45 and 12:00-14:45 UTC (closing 07:15-10:00, 12:15-15:00)
Q = pd.Timedelta(minutes=15)


# ---------------------------------------------------------------------------------------------------- helpers
def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV bars aggregated to `rule` (labels = bar open, UTC boundaries). Incomplete bars are dropped."""
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    n = df["close"].resample(rule, label="left", closed="left").count()
    full = n == n.max()
    return agg[full & agg["close"].notna()]


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Mean true range of the last n bars, the current one included."""
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    """Wilder RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out = out.mask((dn == 0) & (up > 0), 100.0)
    return out.mask((dn == 0) & (up == 0), 50.0)


def known_at_close(series: pd.Series, bar: pd.Timedelta, target_index: pd.DatetimeIndex,
                   target_bar: pd.Timedelta) -> pd.Series:
    """Value of `series` (indexed by its bar open) as known at the close of each target bar."""
    s = series.copy()
    s.index = s.index + bar  # available at the close of its bar
    s = s[~s.index.duplicated(keep="last")].sort_index()
    closes = target_index + target_bar
    out = s.reindex(s.index.union(closes)).ffill().reindex(closes)
    out.index = target_index
    return out


def daily_trend(daily_close: pd.Series, index: pd.DatetimeIndex, sma_days: int) -> pd.Series:
    """+1 / -1 / NaN for each bar of `index`: the previous UTC day's close above / below its daily SMA."""
    sma = daily_close.rolling(sma_days, min_periods=sma_days).mean()
    sig = np.sign(daily_close - sma).where(sma.notna())
    prev = sig.shift(1, freq="1D")  # day d's value is used during day d+1
    return pd.Series(prev.reindex(index.normalize()).to_numpy(), index=index)


def day_last_bar(index: pd.DatetimeIndex) -> np.ndarray:
    """Position of the last bar of each bar's UTC day."""
    pos = pd.Series(np.arange(len(index)), index=index)
    return pos.groupby(index.normalize()).transform("max").to_numpy()


# ------------------------------------------------------------------------------------------------ trade machine
def trade_machine(c, h, l, entry, stop, target=None, deadline=None, exit_long=None, exit_short=None,
                  reverse: bool = False, group=None, per_group: str | None = None,
                  risk: float = RISK_PER_TRADE, cap: float = MAX_WEIGHT):
    """Weights (NaN = hold, 0 = exit at next open) and stop levels from per-bar entry decisions.

    entry[i] in {-1, 0, 1} is decided at the close of bar i with stop[i] (required), target[i] (checked on
    closes) and deadline[i] (position of the bar whose close ends the trade). One trade at a time; signals during
    a trade are ignored. With `reverse`, an opposite signal closes the trade and opens the new one at the same
    next open. `group` (one id per bar, e.g. the day) with `per_group` "side" or "any" allows at most one trade
    per group and side, or per group. Size: `risk` of equity to the stop, at most `cap`.
    """
    T = len(c)
    target = np.full(T, np.nan) if target is None else np.asarray(target, float)
    deadline = np.full(T, -1) if deadline is None else np.asarray(deadline)
    exit_long = np.zeros(T, bool) if exit_long is None else np.asarray(exit_long, bool)
    exit_short = np.zeros(T, bool) if exit_short is None else np.asarray(exit_short, bool)
    w, st = np.full(T, np.nan), np.full(T, np.nan)
    pos, s_lvl, tgt, dl = 0, np.nan, np.nan, -1
    used: set = set()

    def try_enter(i):
        d, s = entry[i], stop[i]
        if d == 0 or np.isnan(s) or np.isnan(c[i]):
            return None
        dist = (c[i] - s) / c[i] * d
        if dist <= 0:
            return None
        key = None
        if per_group is not None:
            key = (group[i], d) if per_group == "side" else group[i]
            if key in used:
                return None
        return d, s, min(risk / dist, cap), key

    def enter(i, e):
        nonlocal pos, s_lvl, tgt, dl
        pos, s_lvl = int(e[0]), e[1]
        w[i], st[i] = e[0] * e[2], s_lvl
        tgt, dl = target[i], deadline[i]
        if e[3] is not None:
            used.add(e[3])

    for i in range(T):
        if np.isnan(c[i]):
            pos = 0  # the engine closes the position on a bar without data
            continue
        if pos != 0:
            if (pos > 0 and l[i] <= s_lvl) or (pos < 0 and h[i] >= s_lvl):
                pos = 0  # the engine fills the stop inside this bar
                e = try_enter(i)
                if e:
                    enter(i, e)
                else:
                    w[i] = 0.0  # explicit flat target, so the engine and this machine stay in step
                continue
            if reverse and entry[i] == -pos:
                e = try_enter(i)
                if e:
                    enter(i, e)
                    continue
            hit = not np.isnan(tgt) and ((pos > 0 and c[i] >= tgt) or (pos < 0 and c[i] <= tgt))
            if hit or (dl >= 0 and i >= dl) or (pos > 0 and exit_long[i]) or (pos < 0 and exit_short[i]):
                w[i], pos = 0.0, 0
            else:
                st[i] = s_lvl
            continue
        e = try_enter(i)
        if e:
            enter(i, e)
    return w, st


def to_15m(w: np.ndarray, st: np.ndarray, index: pd.DatetimeIndex, bar: pd.Timedelta,
           index15: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Place decisions taken at the close of coarser bars on the last 15m bar ending at that close. The stop
    set at a coarse close covers every 15m bar of the next coarse bar."""
    at = index + bar - Q
    ws = pd.Series(w, index=at).reindex(index15)
    per = int(bar / Q)
    ss = pd.Series(st, index=at).reindex(index15).ffill(limit=per - 1)
    return ws, ss


# ------------------------------------------------------------------------------------------------- R3A / R3B
def _sweeps(c, h, l, hi_lvl, lo_lvl, M, allowed, days):
    """Sweep-and-reclaim signals against per-bar levels: a bar trades beyond the level and that bar or one of the
    next M-1 bars closes back inside. Stop = the extreme since the first breach. Returns entry, stop arrays."""
    T = len(c)
    entry, stop = np.zeros(T), np.full(T, np.nan)
    st_hi = st_lo = None  # [start index, extreme]
    armed_hi = armed_lo = True
    cur_day = None
    for i in range(T):
        if days[i] != cur_day:
            cur_day, st_hi, st_lo = days[i], None, None
            armed_hi = armed_lo = True
        H, L = hi_lvl[i], lo_lvl[i]
        if np.isnan(c[i]) or np.isnan(H) or np.isnan(L):
            continue
        sig_s = sig_l = False
        s_ext = l_ext = np.nan
        if st_hi is None and armed_hi and h[i] > H:
            st_hi, armed_hi = [i, h[i]], False
        if st_hi is not None:
            st_hi[1] = max(st_hi[1], h[i])
            if c[i] < H:
                sig_s, s_ext = True, st_hi[1]
                st_hi, armed_hi = None, True
            elif i - st_hi[0] >= M - 1:
                st_hi = None  # expired: re-armed only after a close back below the level
        elif not armed_hi and c[i] < H:
            armed_hi = True
        if st_lo is None and armed_lo and l[i] < L:
            st_lo, armed_lo = [i, l[i]], False
        if st_lo is not None:
            st_lo[1] = min(st_lo[1], l[i])
            if c[i] > L:
                sig_l, l_ext = True, st_lo[1]
                st_lo, armed_lo = None, True
            elif i - st_lo[0] >= M - 1:
                st_lo = None
        elif not armed_lo and c[i] > L:
            armed_lo = True
        if (sig_s and sig_l) or not allowed[i]:
            continue  # both sides at once is ambiguous
        if sig_s:
            entry[i], stop[i] = -1, s_ext
        elif sig_l:
            entry[i], stop[i] = 1, l_ext
    return entry, stop


def _targets(entry, stop, c, mode, ref=None):
    """2R from the decision price, or a reference level; NaN when the target is on the wrong side."""
    t = np.full(len(c), np.nan)
    on = entry != 0
    if mode == "2R":
        t[on] = c[on] + 2 * (c[on] - stop[on])
    else:
        t[on] = ref[on]
    bad = on & ~np.isnan(t) & (((entry > 0) & (t <= c)) | ((entry < 0) & (t >= c)))
    t[bad] = np.nan
    return t


def r3a_pdh_sweep(df: pd.DataFrame, M: int, target: str, killzones: bool):
    """Sweep of the previous UTC day's high/low; at most one trade per side per day; exit by the day's end."""
    idx = df.index
    day = idx.normalize()
    dh = df["high"].groupby(day).max().shift(1)
    dl = df["low"].groupby(day).min().shift(1)
    pdh, pdl = dh.reindex(day).to_numpy(), dl.reindex(day).to_numpy()
    allowed = np.isin(idx.hour, KILLZONE_HOURS) if killzones else np.ones(len(idx), bool)
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    days = day.asi8
    entry, stop = _sweeps(c, h, l, pdh, pdl, M, allowed, days)
    tgt = _targets(entry, stop, c, target, ref=(pdh + pdl) / 2)
    entry = np.where(np.isnan(tgt), 0, entry)
    return trade_machine(c, h, l, entry, stop, tgt, day_last_bar(idx), group=days, per_group="side")


def r3b_asia_sweep(df: pd.DataFrame, M: int, target: str):
    """Sweep of the 00:00-06:00 UTC range inside 07:00-10:00 UTC; one trade per day; exit by 16:00 UTC."""
    idx = df.index
    day = idx.normalize()
    asia = df[idx.hour < 6]
    ah = asia["high"].groupby(asia.index.normalize()).max()
    al = asia["low"].groupby(asia.index.normalize()).min()
    in_lon = (idx.hour >= 7) & (idx.hour < 10)
    hi = np.where(in_lon, ah.reindex(day).to_numpy(), np.nan)  # the range is complete at 06:00, before 07:00
    lo = np.where(in_lon, al.reindex(day).to_numpy(), np.nan)
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    days = day.asi8
    entry, stop = _sweeps(c, h, l, hi, lo, M, np.ones(len(idx), bool), days)
    opp = np.where(entry < 0, lo, np.where(entry > 0, hi, np.nan))
    tgt = _targets(entry, stop, c, target, ref=opp)
    entry = np.where(np.isnan(tgt), 0, entry)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    cut = pos[(idx.hour == 15) & (idx.minute == 45)]  # closes at 16:00 UTC
    dl = pd.Series(cut.to_numpy(), index=cut.index.normalize()).reindex(day).fillna(-1).astype(int).to_numpy()
    return trade_machine(c, h, l, entry, stop, tgt, dl, group=days, per_group="any")


# --------------------------------------------------------------------------------------------------------- R3C
def r3c_fvg(df: pd.DataFrame, trend: pd.Series | None):
    """Fair value gap retest on the bars of `df`; `trend` (+1/-1/NaN per bar) filters direction when given."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df).to_numpy()
    T = len(c)
    tr = None if trend is None else trend.to_numpy()
    entry, stop, tgt, dl = np.zeros(T), np.full(T, np.nan), np.full(T, np.nan), np.full(T, -1)
    gaps = []  # [dir, bottom, top, last valid bar]
    for i in range(T):
        if np.isnan(c[i]):
            gaps = []
            continue
        fired = False
        keep = []
        for g in gaps:  # gaps formed on earlier bars
            d, bot, top, exp = g
            if i > exp:
                continue
            if d > 0:
                if c[i] < bot:
                    continue  # closed through the gap: invalid
                if l[i] <= top and not fired:
                    if tr is None or tr[i] > 0:
                        s = min(bot, l[i])
                        entry[i], stop[i], tgt[i], dl[i] = 1, s, c[i] + 2 * (c[i] - s), i + 24
                        fired = True
                    continue  # the first retest consumes the gap
            else:
                if c[i] > top:
                    continue
                if h[i] >= bot and not fired:
                    if tr is None or tr[i] < 0:
                        s = max(top, h[i])
                        entry[i], stop[i], tgt[i], dl[i] = -1, s, c[i] - 2 * (s - c[i]), i + 24
                        fired = True
                    continue
            keep.append(g)
        gaps = keep
        # a gap formed on this bar (bars i-2, i-1, i) with a displacement body on i-1
        if i >= 2 and not np.isnan(a[i - 1]) and abs(c[i - 1] - o[i - 1]) > a[i - 1]:
            if l[i] > h[i - 2]:
                gaps.append([1, h[i - 2], l[i], i + 24])
            elif h[i] < l[i - 2]:
                gaps.append([-1, h[i], l[i - 2], i + 24])
    return trade_machine(c, h, l, entry, stop, tgt, dl)


# --------------------------------------------------------------------------------------------------------- R3D
def r3d_structure(df: pd.DataFrame, k: int):
    """Break of structure: the first close beyond a confirmed swing level that has not been broken yet. A swing
    at bar i (extreme of bars i-k..i+k) is confirmed at the close of bar i+k. Opposite signals reverse."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    T = len(c)
    entry, stop = np.zeros(T), np.full(T, np.nan)
    last_hi = last_lo = np.nan
    hi_broken = lo_broken = True
    for j in range(T):
        i = j - k
        if i - k >= 0:
            win_h, win_l = h[i - k:j + 1], l[i - k:j + 1]
            if not np.isnan(win_h).any() and h[i] == win_h.max():
                last_hi, hi_broken = h[i], False
            if not np.isnan(win_l).any() and l[i] == win_l.min():
                last_lo, lo_broken = l[i], False
        if np.isnan(c[j]) or np.isnan(last_hi) or np.isnan(last_lo):
            continue
        if not hi_broken and c[j] > last_hi:
            entry[j], stop[j], hi_broken = 1, last_lo, True
        elif not lo_broken and c[j] < last_lo:
            entry[j], stop[j], lo_broken = -1, last_hi, True
    return trade_machine(c, h, l, entry, stop, reverse=True)


# --------------------------------------------------------------------------------------------------------- R3E
def session_keys(index: pd.DatetimeIndex, session: str) -> tuple[np.ndarray, np.ndarray]:
    """Per bar: session id (or -1 outside a session) and position of the session's last bar.
    "utc": the UTC day. "ny": New York exchange hours, weekdays 09:30-16:00 local time (DST-aware)."""
    if session == "utc":
        key = index.normalize().asi8
    else:
        local = index.tz_convert(NY)
        mins = local.hour * 60 + local.minute
        in_sess = (local.dayofweek < 5) & (mins >= 570) & (mins < 960)
        key = np.where(in_sess, local.normalize().tz_localize(None).asi8, -1)
    pos = pd.Series(np.arange(len(index)))
    end = pos.groupby(key).transform("max").to_numpy()
    return key, np.where(key != -1, end, -1)


def r3e_orb(df: pd.DataFrame, session: str, minutes: int):
    """Opening range breakout: first close beyond the first `minutes` of the session; one trade per session;
    stop at the opposite edge; exit at the session's end."""
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    T = len(c)
    nbar = minutes // 15
    key, end_pos = session_keys(df.index, session)
    entry, stop = np.zeros(T), np.full(T, np.nan)
    cur, k, orh, orl, done = None, 0, -np.inf, np.inf, False
    for i in range(T):
        if key[i] == -1:
            continue
        if key[i] != cur:
            cur, k, orh, orl, done = key[i], 0, -np.inf, np.inf, False
        k += 1
        if np.isnan(c[i]):
            done = True  # a missing bar inside the session: no trade in it
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
    return trade_machine(c, h, l, entry, stop, deadline=end_pos)


# --------------------------------------------------------------------------------------------------- R3F / R3G
def protective_stop_dist(h1: pd.DataFrame, index15: pd.DatetimeIndex) -> np.ndarray:
    """3 x ATR(14) of 1h bars, as known at the close of each 15m bar."""
    a = known_at_close(atr(h1), pd.Timedelta(hours=1), index15, Q)
    return 3 * a.to_numpy()


def r3f_intraday_momentum(df: pd.DataFrame, h1: pd.DataFrame, hour: int):
    """At `hour`:00 UTC, trade in the direction of the return since the UTC day's open, until the day's end."""
    idx = df.index
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    day_open = df["open"].groupby(idx.normalize()).transform("first").to_numpy()
    at = (idx.hour == hour - 1) & (idx.minute == 45)  # the bar that closes at hour:00
    d = np.sign(c / day_open - 1.0)
    entry = np.where(at, d, 0.0)
    sd = protective_stop_dist(h1, idx)
    stop = np.where(entry != 0, c - entry * sd, np.nan)
    return trade_machine(c, h, l, entry, stop, deadline=day_last_bar(idx))


def r3g_vwap(df: pd.DataFrame, h1: pd.DataFrame, K: float):
    """Fade a K-sigma deviation from the session VWAP (from 00:00 UTC); exit on a close across the VWAP."""
    idx = df.index
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    day = idx.normalize()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    vwap = (pv / vv.replace(0, np.nan)).to_numpy()
    d = pd.Series(c / vwap - 1.0, index=idx)
    sig = d.rolling(30 * 96, min_periods=30 * 96).std().shift(1)  # the previous 30 days
    z = (d / sig).to_numpy()
    window = (idx.hour >= 2) & (idx.hour < 22)
    entry = np.where(window & (z > K), -1.0, np.where(window & (z < -K), 1.0, 0.0))
    sd = protective_stop_dist(h1, idx)
    stop = np.where(entry != 0, c - entry * sd, np.nan)
    return trade_machine(c, h, l, entry, stop, deadline=day_last_bar(idx), exit_long=c >= vwap, exit_short=c <= vwap)


# --------------------------------------------------------------------------------------------------------- R3H
def r3h_pullback(df4: pd.DataFrame, trend: pd.Series, T_: float):
    """RSI(2) pullback in the direction of the daily trend on 4h bars; exit on RSI(2) > 70 (< 30) or after 30 bars."""
    c, h, l = df4["close"].to_numpy(), df4["high"].to_numpy(), df4["low"].to_numpy()
    r = rsi(df4["close"]).to_numpy()
    tr = trend.to_numpy()
    entry = np.where((tr > 0) & (r < T_), 1.0, np.where((tr < 0) & (r > 100 - T_), -1.0, 0.0))
    a = atr(df4).to_numpy()
    stop = np.where(entry != 0, c - entry * 3 * a, np.nan)
    dl = np.where(entry != 0, np.arange(len(c)) + 30, -1)
    return trade_machine(c, h, l, entry, stop, deadline=dl, exit_long=r > 70, exit_short=r < 30)
