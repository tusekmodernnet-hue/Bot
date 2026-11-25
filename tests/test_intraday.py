"""Round 3 intraday families: trade mechanics on hand-built cases and no look-ahead on synthetic bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest import engine
from cqs.backtest.costs import CostModel
from cqs.strategies import intraday as I


# ------------------------------------------------------------------------------------------- trade machine
def _flat(n=8, px=100.0):
    c = np.full(n, px)
    return c, c + 1, c - 1


def test_trade_machine_sizes_by_risk_and_exits_on_stop():
    c, h, l = _flat()
    l[2] = 94.0  # stop at 95 is hit inside bar 2
    entry = np.zeros(8)
    entry[0] = 1
    stop = np.full(8, np.nan)
    stop[0] = 95.0
    w, st = I.trade_machine(c, h, l, entry, stop)
    assert w[0] == pytest.approx(0.005 / 0.05)  # 0.5% risk over a 5% stop distance
    assert st[0] == st[1] == 95.0 and np.isnan(w[1])
    assert w[2] == 0.0 and np.isnan(st[2])  # explicit flat target after the stop
    assert np.isnan(w[3:]).all()


def test_trade_machine_target_deadline_cap_and_signals_during_trade():
    c, h, l = _flat()
    l = c - 0.05  # a tight range, so the 99.9 stop is never touched
    c[3] = 105.0
    h[3] = 106.0
    entry = np.zeros(8)
    entry[[0, 1]] = 1  # the second signal arrives during the trade and is ignored
    stop = np.full(8, 99.9)
    target = np.full(8, 104.0)
    w, st = I.trade_machine(c, h, l, entry, stop, target=target)
    assert w[0] == 1.0  # 0.1% stop distance: capped at 1x
    assert np.isnan(w[1]) and w[3] == 0.0  # target on the close of bar 3, exit at the next open
    w, _ = I.trade_machine(c, h, l, entry, stop, deadline=np.full(8, 2))
    assert w[2] == 0.0 and np.isnan(w[1])


def test_trade_machine_reverse_and_group_limits():
    c, h, l = _flat()
    entry = np.array([1, 0, -1, 0, 1, 0, -1, 0], float)
    stop = np.where(entry > 0, 95.0, 105.0)
    w, st = I.trade_machine(c, h, l, entry, stop, reverse=True)
    assert w[0] > 0 and w[2] < 0 and w[4] > 0 and w[6] < 0
    assert st[2] == 105.0 and st[3] == 105.0
    # one trade per side per group: every trade ends at its deadline (next bar)
    dl = np.arange(8) + 1
    grp = np.zeros(8, int)
    w, _ = I.trade_machine(c, h, l, entry, stop, deadline=dl, group=grp, per_group="side")
    assert w[0] > 0 and w[2] < 0 and np.isnan(w[4]) and np.isnan(w[6])
    w, _ = I.trade_machine(c, h, l, entry, stop, deadline=dl, group=grp, per_group="any")
    assert w[0] > 0 and np.isnan(w[2]) and np.isnan(w[4])


def test_to_15m_places_decisions_on_the_last_quarter_and_carries_stops():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    idx15 = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    ws, ss = I.to_15m(np.array([0.5, np.nan, 0.0]), np.array([90.0, 91.0, np.nan]), idx, pd.Timedelta(hours=1), idx15)
    assert ws.iloc[3] == 0.5 and ws.iloc[11] == 0.0 and ws.drop(ws.index[[3, 11]]).isna().all()
    assert (ss.iloc[3:7] == 90.0).all() and (ss.iloc[7:11] == 91.0).all()
    assert ss.iloc[:3].isna().all() and np.isnan(ss.iloc[11])


def test_ny_session_follows_daylight_saving():
    idx = pd.date_range("2024-03-07", "2024-03-13", freq="15min", tz="UTC", inclusive="left")
    key, end = I.session_keys(idx, "ny")
    first = pd.Series(idx[key != -1]).groupby(key[key != -1]).first()
    hours = sorted({(t.hour, t.minute) for t in first})
    assert hours == [(13, 30), (14, 30)]  # 09:30 New York in summer and winter time
    assert len(first) == 4  # Thu, Fri, Mon, Tue (no weekend sessions)
    last = idx[end[key != -1]]
    assert {(t.hour, t.minute) for t in last} == {(19, 45), (20, 45)}  # 15:45 local, closes at 16:00


# ------------------------------------------------------------------------------------- synthetic market
def _bars15(days=120, seed=3, start="2021-01-01"):
    rng = np.random.default_rng(seed)
    n = days * 96
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    c = 30000 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0015, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0015, n)))
    v = rng.uniform(50, 150, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def _perturb_after(df, t, seed=99):
    """Same calendar, different prices for every bar opening after t."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = out.index > t
    k = m.sum()
    f = np.exp(np.cumsum(rng.normal(0, 0.01, k)))
    for col in ("open", "high", "low", "close"):
        out.loc[m, col] = out.loc[m, col].to_numpy() * f * 1.07
    out.loc[m, "volume"] = rng.uniform(1, 500, k)
    return out


def _all_variants(df15):
    h1 = I.resample(df15, "1h")
    h4 = I.resample(h1, "4h")
    dclose = df15["close"].resample("1D").last()
    idx = df15.index
    hr, fr = pd.Timedelta(hours=1), pd.Timedelta(hours=4)
    out = {}
    for M in (1, 4):
        for tg in ("2R", "mid"):
            for kz in (False, True):
                out[f"A{M}{tg}{kz}"] = I.r3a_pdh_sweep(df15, M, tg, kz)
    for M in (1, 2):
        for tg in ("opposite", "2R"):
            out[f"B{M}{tg}"] = I.r3b_asia_sweep(df15, M, tg)
    out["C15"] = I.r3c_fvg(df15, None)
    out["C15t"] = I.r3c_fvg(df15, I.daily_trend(dclose, idx, 5))
    out["C1ht"] = I.to_15m(*I.r3c_fvg(h1, I.daily_trend(dclose, h1.index, 5)), h1.index, hr, idx)
    out["D1h"] = I.to_15m(*I.r3d_structure(h1, 3), h1.index, hr, idx)
    out["D4h"] = I.to_15m(*I.r3d_structure(h4, 3), h4.index, fr, idx)
    for s in ("utc", "ny"):
        for m in (15, 60):
            out[f"E{s}{m}"] = I.r3e_orb(df15, s, m)
    for H in (12, 20):
        out[f"F{H}"] = I.r3f_intraday_momentum(df15, h1, H)
    out["G2"] = I.r3g_vwap(df15, h1, 2.0)
    out["H10"] = I.to_15m(*I.r3h_pullback(h4, I.daily_trend(dclose, h4.index, 5), 10), h4.index, fr, idx)
    return {k: tuple(np.asarray(x, float) for x in v) for k, v in out.items()}


def test_round3_families_use_no_future_data():
    df = _bars15()
    full = _all_variants(df)
    for k, (w, _) in full.items():
        assert np.nansum(np.abs(w)) > 0, f"{k} never trades on synthetic data"
    for pos in (5003, 6001, 7202, 8410):  # every quarter-hour offset, different times of day
        t = df.index[pos]
        cut = _all_variants(_perturb_after(df, t))
        for k in full:
            for a, b in zip(full[k], cut[k]):
                assert np.array_equal(a[:pos + 1], b[:pos + 1], equal_nan=True), f"{k} looks ahead at {t}"


def test_engine_executes_every_machine_trade():
    df = _bars15(days=60, seed=5)
    h1 = I.resample(df, "1h")
    costs = CostModel(default_slippage_bps=2.0)
    cases = {"A": I.r3a_pdh_sweep(df, 4, "2R", False), "E": I.r3e_orb(df, "utc", 30), "C": I.r3c_fvg(df, None),
             "D": I.to_15m(*I.r3d_structure(h1, 3), h1.index, pd.Timedelta(hours=1), df.index),
             "G": I.r3g_vwap(df, h1, 2.0)}
    for k, (w, st) in cases.items():
        w = pd.Series(np.asarray(w, float), index=df.index)
        st = pd.Series(np.asarray(st, float), index=df.index)
        frame = lambda col: df[[col]].rename(columns={col: "X"})  # noqa: E731
        res = engine.run(frame("open"), frame("high"), frame("low"), frame("close"), w.to_frame("X"), costs,
                         stops=st.to_frame("X"), reenter_after_stop=True)
        n_entries = int(((w != 0) & w.notna()).sum())
        assert n_entries > 0
        assert len(res.trades) == n_entries, k


def test_helpers_use_no_future_data():
    df = _bars15(days=40, seed=8)
    dclose = df["close"].resample("1D").last()
    for pos in range(1500, 1600, 7):  # every quarter-hour offset and several hours
        t = df.index[pos]
        d2 = _perturb_after(df, t)
        a = I.protective_stop_dist(I.resample(df, "1h"), df.index)
        b = I.protective_stop_dist(I.resample(d2, "1h"), df.index)
        assert np.array_equal(a[:pos + 1], b[:pos + 1], equal_nan=True)
        a = I.daily_trend(dclose, df.index, 5).to_numpy()
        b = I.daily_trend(d2["close"].resample("1D").last(), df.index, 5).to_numpy()
        assert np.array_equal(a[:pos + 1], b[:pos + 1], equal_nan=True)
