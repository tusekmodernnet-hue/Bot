"""Round 5 order-flow families: definitions on hand-built cases and no look-ahead on synthetic bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest.limit_sim import Fees, simulate
from cqs.strategies import intraday as I
from cqs.strategies import orderflow as F

Q = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)


# ------------------------------------------------------------------------------------------- definitions
def test_flow_is_the_net_aggressor_share_of_volume():
    df = pd.DataFrame({"volume": [10.0, 20.0, 0.0, 10.0], "taker_buy_volume": [8.0, 5.0, 0.0, 5.0]})
    f1 = F.flow(df).to_numpy()
    assert f1[0] == pytest.approx(0.6) and f1[1] == pytest.approx(-0.5) and np.isnan(f1[2]) and f1[3] == 0.0
    f2 = F.flow(df, 2).to_numpy()
    assert np.isnan(f2[0]) and f2[1] == pytest.approx((6.0 - 10.0) / 30.0)  # sums, not a mean of ratios


def test_oi_at_close_uses_available_time_and_skips_empty_snapshots():
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    create = pd.date_range(t0, periods=12, freq="5min")
    oi = pd.DataFrame({"create_time": create, "available_time": create + pd.Timedelta(minutes=5),
                       "sum_open_interest": np.arange(100.0, 112.0)})
    oi.loc[5, "sum_open_interest"] = 0.0  # empty record (created 00:25, available 00:30)
    idx = pd.date_range(t0 - Q, periods=5, freq="15min")  # bars closing 00:00, 00:15, ..., 01:00
    out = F.oi_at_close(oi, idx, Q)
    assert np.isnan(out.iloc[0])  # nothing available yet at 00:00
    assert out.iloc[1] == 102.0  # at 00:15 the latest available snapshot was created at 00:10
    assert out.iloc[2] == 104.0  # at 00:30 the 00:25 record is empty: the 00:20 snapshot stands
    assert out.iloc[3] == 108.0 and out.iloc[4] == 111.0


def _day_bars(days=2, px=100.0, start="2024-01-01"):
    idx = pd.date_range(start, periods=96 * days, freq="15min", tz="UTC")
    return pd.DataFrame({"open": px, "high": px + 0.1, "low": px - 0.1, "close": px, "volume": 10.0,
                         "taker_buy_volume": 5.0}, index=idx)


def test_sweep_absorption_needs_aggressors_on_the_side_of_the_breach():
    df = _day_bars()
    df.loc[df.index[30], "high"] = 105.0  # day 1 range 95-105
    df.loc[df.index[60], "low"] = 95.0
    k = 96 + 40  # day 2, 10:00 UTC: breach, still above, close back below the 105 high
    for j, (hi, cl) in enumerate([(106.0, 105.5), (105.8, 105.2), (105.6, 104.5)]):
        df.loc[df.index[k + j], ["high", "close"]] = [hi, cl]
    starts = np.full(len(df), -1)
    day = df.index.normalize()
    pdh = df["high"].groupby(day).max().shift(1).reindex(day).to_numpy()
    pdl = df["low"].groupby(day).min().shift(1).reindex(day).to_numpy()
    I._sweeps(df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), pdh, pdl, 4,
              np.ones(len(df), bool), day.asi8, starts=starts)
    assert starts[k + 2] == k and (np.delete(starts, k + 2) == -1).all()

    buyers = df.copy()
    buyers.loc[buyers.index[k:k + 3], "taker_buy_volume"] = 8.0  # I = 0.6 on the sweep bars
    w, st = F.r5c_sweep_absorption(buyers, theta=0.1)
    assert w[k + 2] < 0 and st[k + 2] == 106.0 and np.nansum(np.abs(w[:k + 2])) == 0
    assert w[k + 3] == 0.0  # target = middle of the previous day's range (100) reached on the next close

    early = df.copy()  # buyers only before the breach and after the reclaim: the sweep bars are neutral
    early.loc[early.index[k - 1], "taker_buy_volume"] = 10.0
    early.loc[early.index[k + 3], "taker_buy_volume"] = 10.0
    w, _ = F.r5c_sweep_absorption(early, theta=0.0)
    assert np.nansum(np.abs(w)) == 0


class _AllSessions(dict):
    def get(self, k, default=None):
        return True


def _orb_day():
    """Wednesday 2024-01-03 (New York 09:30 = 14:30 UTC): 30-minute range 99-101, close above it at 15:15 UTC."""
    df = _day_bars(days=1, start="2024-01-03")
    df.loc[df.index[58], ["high", "low"]] = [101.0, 99.0]
    df.loc[df.index[60], ["high", "close"]] = [101.8, 101.5]
    return df


def test_orb_limit_plan_rests_at_the_broken_edge_until_noon(monkeypatch):
    monkeypatch.setattr(F, "ny_vol_filter", lambda m15, h1, key: _AllSessions())
    df = _orb_day()
    idx1 = pd.date_range("2024-01-03", periods=1440, freq="1min", tz="UTC")
    p = F.r5d_orb_limit_plans(df, None, idx1, 30)
    assert len(p) == 1
    assert p.start[0] == 15 * 60 + 15 and p.end[0] == 16 * 60 + 59  # from 10:15 to 11:59 New York
    assert p.deadline_abs[0] == 20 * 60 + 59 and p.deadline_rel[0] == -1  # time exit at the 16:00 open
    assert p.buy_px[0] == 101.0 and p.buy_sl[0] == 99.0 and np.isnan(p.buy_tp[0]) and np.isnan(p.sell_px[0])

    o = np.full(1440, 101.5)
    h, l, c = o + 0.1, o - 0.1, o.copy()
    l[960] = 100.9  # 11:00 New York: trades through the edge
    _, t = simulate(idx1, o, h, l, c, p, Fees())
    assert len(t) == 1 and t["entry"].iloc[0] == 101.0 and t["reason"].iloc[0] == "time"
    assert t["entry_time"].iloc[0] == idx1[960] and t["exit_time"].iloc[0] == pd.Timestamp("2024-01-03 21:00", tz="UTC")

    l2 = o - 0.1
    l2[1080] = 100.9  # 13:00 New York: the order has expired at noon
    _, t2 = simulate(idx1, o, h, l2, c, p, Fees())
    assert len(t2) == 0

    late = _orb_day()
    late.loc[late.index[60], ["high", "close"]] = [100.2, 100.0]
    late.loc[late.index[68], ["high", "close"]] = [101.8, 101.5]  # closes above the range at 12:15 New York
    assert len(F.r5d_orb_limit_plans(late, None, idx1, 30)) == 0


def test_orb_market_trades_the_first_close_beyond_the_range(monkeypatch):
    monkeypatch.setattr(F, "ny_vol_filter", lambda m15, h1, key: _AllSessions())
    w, st = F.r5d_orb_market(_orb_day(), None, 30)
    assert w[60] > 0 and st[60] == 99.0 and np.nansum(np.abs(w[:60])) == 0
    assert w[83] == 0.0  # the bar closing at 16:00 New York ends the trade


def test_volatility_filter_compares_with_the_previous_60_sessions():
    idx = pd.date_range("2024-01-01", periods=120 * 96, freq="15min", tz="UTC")
    c = np.full(len(idx), 100.0)
    rng = np.random.default_rng(1)
    width = np.repeat(rng.uniform(0.1, 2.0, 120), 96)  # the bar range changes once a day
    df = pd.DataFrame({"open": c, "high": c + width, "low": c - width, "close": c, "volume": 1.0}, index=idx)
    h1 = I.resample(df, "1h")
    key, _ = I.session_keys(idx, "ny")
    ok = F.ny_vol_filter(df, h1, key)
    a = (I.known_at_close(I.atr(h1), H1, idx, Q) / df["close"]).to_numpy()
    sess = sorted(ok)
    first = {s: int(np.flatnonzero(key == s)[0]) for s in sess}
    vals = np.array([a[first[s] - 1] for s in sess])
    for j, s in enumerate(sess):
        expected = j >= 60 and vals[j] >= np.median(vals[j - 60:j])
        assert ok[s] == expected


# ------------------------------------------------------------------------------------- synthetic market
def _bars15_flow(days=130, seed=11, start="2021-03-01"):
    rng = np.random.default_rng(seed)
    n = days * 96
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    r = 0.003 * rng.standard_t(3, n) / np.sqrt(3)  # fat tails, so that flush bars occur
    c = 30000 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0015, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0015, n)))
    v = rng.uniform(50, 150, n)
    share = np.clip(0.5 + 40 * r + rng.normal(0, 0.1, n), 0.02, 0.98)  # aggressors lean with the move
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "taker_buy_volume": v * share},
                        index=idx)


def _oi(df, seed=12):
    rng = np.random.default_rng(seed)
    t = pd.date_range(df.index[0], df.index[-1] + Q, freq="5min")
    v = 1e5 * np.exp(np.cumsum(rng.normal(0, 0.006, len(t))))
    v[::97] = 0.0  # empty records
    return pd.DataFrame({"create_time": t, "available_time": t + pd.Timedelta(minutes=5), "sum_open_interest": v})


def _perturb_after(df, oi, t, seed=99):
    """Same calendar; new bars (prices, intrabar returns, ranges, volume, aggressor volume) for every bar opening
    after t, and different open interest for every snapshot not yet available at the close of the bar at t."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = out.index > t
    k = int(m.sum())
    c0 = float(out.loc[~m, "close"].iloc[-1])
    c = c0 * 1.07 * np.exp(np.cumsum(rng.normal(0, 0.01, k)))
    o = np.r_[c0 * 1.03, c[:-1]] * np.exp(rng.normal(0, 0.004, k))  # gaps: each bar's own return changes too
    out.loc[m, "open"], out.loc[m, "close"] = o, c
    out.loc[m, "high"] = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, k)))
    out.loc[m, "low"] = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, k)))
    v = rng.uniform(1, 500, k)
    out.loc[m, "volume"] = v
    out.loc[m, "taker_buy_volume"] = v * rng.uniform(0, 1, k)
    oi2 = oi.copy()
    late = (oi2["available_time"] > t + Q).to_numpy()
    oi2.loc[late, "sum_open_interest"] = oi2.loc[late, "sum_open_interest"].to_numpy() * rng.uniform(0.9, 1.1, late.sum())
    return out, oi2


def _h1(df):
    h1 = I.resample(df, "1h")
    h1["taker_buy_volume"] = df["taker_buy_volume"].resample("1h").sum().reindex(h1.index)
    return h1


def _round5(df, oi, idx1):
    h1 = _h1(df)
    out = {}
    for L in (1, 4):
        out[f"A{L}"] = I.to_15m(*F.r5a_flow(h1, L), h1.index, H1, df.index)
    for k in (3.0, 4.0):
        out[f"B{k}"] = F.r5b_flush(df, h1, oi, k)
    for th in (0.0, 0.1):
        out[f"C{th}"] = F.r5c_sweep_absorption(df, th)
    for m in (15, 30):
        out[f"Dm{m}"] = F.r5d_orb_market(df, h1, m)
    out = {k: tuple(np.asarray(x, float) for x in v) for k, v in out.items()}
    plans = {m: F.r5d_orb_limit_plans(df, h1, idx1, m) for m in (15, 30)}
    return out, plans


def _plans_until(p, idx1, t_end):
    """Plans placed at or before t_end, one row each."""
    keep = np.asarray(p.start) < idx1.searchsorted(t_end, side="right")
    return np.column_stack([np.asarray(getattr(p, f), float)[keep] for f in p.__dataclass_fields__])


def _features(df, oi):
    """The continuous inputs of every rule: a one-bar leak changes them at the cut, unlike sparse decisions."""
    h1 = _h1(df)
    day = df.index.normalize()
    pdh = df["high"].groupby(day).max().shift(1).reindex(day).to_numpy()
    pdl = df["low"].groupby(day).min().shift(1).reindex(day).to_numpy()
    starts = np.full(len(df), -1)
    entry, _ = I._sweeps(df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), pdh, pdl, 4,
                         np.ones(len(df), bool), day.asi8, starts=starts)
    key, _ = I.session_keys(df.index, "ny")
    return {"h1_index": h1.index, "z1": F.r5a_z(h1, 1).to_numpy(), "z4": F.r5a_z(h1, 4).to_numpy(),
            "r5b": F.r5b_features(df, oi), "sweep_flow": F.sweep_flow(df, entry, starts),
            "vol": F.ny_vol_values(df, h1, key)}


def test_round5_features_use_no_future_data():
    df = _bars15_flow()
    oi = _oi(df)
    full = _features(df, oi)
    assert np.isfinite(full["z4"]).sum() > 1000 and full["r5b"].notna().all(axis=1).sum() > 5000
    assert (full["vol"]["median_prev60"].notna()).sum() > 20 and np.isfinite(full["sweep_flow"]).sum() > 50
    cuts = list(np.random.default_rng(5).choice(np.arange(35 * 96, len(df) - 8), 25, replace=False))
    sweeps = np.flatnonzero(np.isfinite(full["sweep_flow"]))
    cuts += list(sweeps[sweeps > 35 * 96][:8])
    opens = full["vol"]["first"][full["vol"]["median_prev60"].notna()].to_numpy()
    cuts += list(opens[:4] - 1)  # the bar closing at 09:30 New York: the filter value is fixed at its close
    for pos in cuts:
        t = df.index[pos]
        cut = _features(*_perturb_after(df, oi, t))
        known_h1 = np.asarray(full["h1_index"] + H1 <= t + Q)
        for k in ("z1", "z4"):
            assert np.array_equal(full[k][known_h1], cut[k][known_h1], equal_nan=True), f"{k} looks ahead at {t}"
        a, b = full["r5b"].loc[:t].to_numpy(), cut["r5b"].loc[:t].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), f"R5B features look ahead at {t}"
        assert np.array_equal(full["sweep_flow"][:pos + 1], cut["sweep_flow"][:pos + 1], equal_nan=True), \
            f"sweep flow looks ahead at {t}"
        v, w = full["vol"], cut["vol"]
        known = (v["first"] - 1 <= pos).to_numpy()
        assert np.array_equal(v[known].to_numpy(float), w[known].to_numpy(float), equal_nan=True), \
            f"volatility filter looks ahead at {t}"


def test_round5_decisions_use_no_future_data():
    df = _bars15_flow()
    oi = _oi(df)
    idx1 = pd.date_range(df.index[0], df.index[-1] + Q, freq="1min", inclusive="left")
    full, plans = _round5(df, oi, idx1)
    cuts = {9001, 10002, 11203, 12010}  # every quarter-hour offset, inside and outside New York hours
    for k, (w, _) in full.items():
        e = np.flatnonzero((w != 0) & ~np.isnan(w))
        assert len(e) > 0, f"{k} never trades on synthetic data"
        cuts.update(e[e > 35 * 96][:3].tolist())  # cut right after decisions: a leak there flips them
    for m, p in plans.items():
        assert len(p) > 0, f"limit ORB {m} places no order on synthetic data"
        dec = df.index.get_indexer(idx1[np.asarray(p.start)] - Q)
        cuts.update(dec[:3].tolist())
    for pos in sorted(cuts):
        t = df.index[pos]
        cut, cut_plans = _round5(*_perturb_after(df, oi, t), idx1)
        for k in full:
            for a, b in zip(full[k], cut[k]):
                assert np.array_equal(a[:pos + 1], b[:pos + 1], equal_nan=True), f"{k} looks ahead at {t}"
        for m in plans:
            a, b = _plans_until(plans[m], idx1, t + Q), _plans_until(cut_plans[m], idx1, t + Q)
            assert a.shape == b.shape and np.array_equal(a, b, equal_nan=True), f"limit ORB {m} looks ahead at {t}"
