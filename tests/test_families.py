import numpy as np
import pandas as pd
import pytest

from cqs.data.universe import UniverseRules, eligible
from cqs.research.panel import Panel
from cqs.strategies import families


def synthetic_panel(days=420, n=24, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=days, freq="1D", tz="UTC")
    cols = ["BTCUSDT", "ETHUSDT"] + [f"A{i}USDT" for i in range(n - 2)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (days, n)), axis=0)), index=idx, columns=cols)
    close.iloc[:150, -1] = np.nan  # a late listing
    close.iloc[300:, -2] = np.nan  # a delisting
    open_ = close.shift(1).fillna(close)
    qv = close.notna() * 1e8
    vol = close.notna() * 1.0
    uni = eligible({"close": close, "quote_volume": qv.where(close.notna()), "volume": vol.where(close.notna())},
                   UniverseRules(min_age_days=30, top_n=None, min_avg_quote_volume=1))
    fidx = pd.date_range("2021-01-01", periods=days * 3, freq="8h", tz="UTC")
    funding = pd.DataFrame(rng.normal(1e-4, 2e-4, (len(fidx), n)), index=fidx, columns=cols)
    return Panel(open_, np.maximum(open_, close), np.minimum(open_, close), close, qv, funding, uni)


def cut_future(p: Panel, t: pd.Timestamp) -> Panel:
    """Same panel, but every value after t is replaced (prices tripled, funding flipped)."""
    c = p.close.copy()
    c.loc[c.index > t] *= 3.0
    f = p.funding.copy()
    f.loc[f.index > t + pd.Timedelta(days=1)] *= -5
    return Panel(p.open, p.high, p.low, c, p.quote_volume, f, p.universe)


@pytest.mark.parametrize("family", list(families.GRIDS))
def test_family_weights_do_not_use_future_data(family):
    p = synthetic_panel()
    t = p.close.index[350]
    fname, params = families.GRIDS[family][0]
    w1 = families.FUNCS[fname](p, **params)
    w2 = families.FUNCS[fname](cut_future(p, t), **params)
    pd.testing.assert_frame_equal(w1.loc[:t], w2.loc[:t])


@pytest.mark.parametrize("family", list(families.GRIDS))
def test_family_weights_respect_caps_and_data(family):
    p = synthetic_panel()
    for fname, params in families.GRIDS[family]:
        w = families.FUNCS[fname](p, **params)
        assert (w.fillna(0).abs().sum(axis=1) <= 2.0 + 1e-9).all()
        alts = [c for c in w.columns if c not in ("BTCUSDT", "ETHUSDT")]
        assert (w[alts].fillna(0).abs() <= 0.25 + 1e-12).all().all()
        assert (w.fillna(0).where(p.close.isna(), 0.0) == 0).all().all()
        assert w.fillna(0).abs().sum().sum() > 0  # the variant actually trades


def test_h6_no_lookahead_and_stop_levels():
    from cqs.strategies.breakouts import false_breakout
    rng = np.random.default_rng(5)
    idx = pd.date_range("2021-01-01", periods=2000, freq="1h", tz="UTC")
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.006, 2000))), index=idx)
    o = c.shift(1).fillna(c)
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, 2000)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, 2000)))
    w1, s1 = false_breakout(o, h, l, c, 24, 3)
    t = 1500
    c2, h2, l2 = c.copy(), h.copy(), l.copy()
    c2.iloc[t + 1:] *= 1.5; h2.iloc[t + 1:] *= 1.5; l2.iloc[t + 1:] *= 1.5
    w2, s2 = false_breakout(o, h2, l2, c2, 24, 3)
    pd.testing.assert_series_equal(w1.iloc[:t + 1], w2.iloc[:t + 1])
    entries = w1[w1.notna() & (w1 != 0)]
    assert len(entries) > 10
    for ts, wt in entries.items():  # short stops sit above the close, long stops below
        assert (s1[ts] > c[ts]) if wt < 0 else (s1[ts] < c[ts])
        assert abs(wt) <= 1.0


def _hourly_assets(k=6, bars=2000, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=bars, freq="1h", tz="UTC")
    cols = ["BTCUSDT"] + [f"A{i}USDT" for i in range(k - 1)]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.006, (bars, k)), axis=0)), index=idx, columns=cols)
    o = c.shift(1).fillna(c)
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, (bars, k))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, (bars, k))))
    return o, h, l, c


def test_h6_entry_gate_blocks_new_trades_only():
    from cqs.strategies.breakouts import h6_weights
    o, h, l, c = _hourly_assets()
    w_all, _ = h6_weights(o, h, l, c, 24, 3)
    gate = pd.DataFrame(True, index=c.index, columns=c.columns)
    gate.iloc[1000:] = False
    w_gated, _ = h6_weights(o, h, l, c, 24, 3, can_enter=gate)
    entries = lambda w: w.notna() & (w != 0)  # noqa: E731
    assert not entries(w_gated).iloc[1000:].any().any()  # no new trade where the gate is closed
    pd.testing.assert_frame_equal(w_gated.iloc[:1000], w_all.iloc[:1000])  # identical while it is open
    # a trade opened just before the gate closed still gets its exit
    last_entry = entries(w_gated).iloc[:1000].stack()
    last_entry = last_entry[last_entry].index[-1]
    after = w_gated.loc[last_entry[0]:, last_entry[1]].iloc[1:]
    assert ((after == 0).any()) or after.notna().sum() == 0  # exit order, or a stop handled by the engine


def test_h6_per_asset_and_gross_caps():
    from cqs.strategies import breakouts
    o, h, l, c = _hourly_assets(k=12)
    caps = {s: (1.0 if s == "BTCUSDT" else 0.25) for s in c.columns}
    w, _ = breakouts.h6_weights(o, h, l, c, 24, 1, max_weight=caps, max_gross=0.6)
    for s in c.columns:
        assert w[s].abs().max() <= caps[s] + 1e-12
    # rebuild the open-trade weights after the cap and check the portfolio never exceeds it
    held, w_raw = {}, {}
    for s in c.columns:
        w_raw[s], _, held[s] = breakouts._false_breakout(h[s], l[s], c[s], 24, 1, None, caps[s])
    raw, w_raw = pd.DataFrame(held, index=c.index), pd.DataFrame(w_raw, index=c.index)
    entry = w_raw.notna() & (w_raw != 0)  # entries before the cap (a capped entry may become 0)
    scale = (w / w_raw).where(entry).ffill().where(raw != 0).fillna(0)
    gross = (raw * scale).abs().sum(axis=1)
    assert gross.max() <= 0.6 + 1e-9
    assert (w.where(entry).abs() < w_raw.where(entry).abs() - 1e-12).any().any()  # the cap binds here


def test_ml_features_use_no_future_data_and_training_is_purged():
    from cqs.research import ml_meta
    p = synthetic_panel()
    pm = Panel(p.open[["BTCUSDT", "ETHUSDT"]], p.high[["BTCUSDT", "ETHUSDT"]], p.low[["BTCUSDT", "ETHUSDT"]],
               p.close[["BTCUSDT", "ETHUSDT"]], p.quote_volume[["BTCUSDT", "ETHUSDT"]],
               p.funding[["BTCUSDT", "ETHUSDT"]], p.close[["BTCUSDT", "ETHUSDT"]].notna())
    t = p.close.index[300]
    f1 = ml_meta.features(pm, p)
    p2 = cut_future(p, t)
    pm2 = Panel(p2.open[["BTCUSDT", "ETHUSDT"]], p2.high[["BTCUSDT", "ETHUSDT"]], p2.low[["BTCUSDT", "ETHUSDT"]],
                p2.close[["BTCUSDT", "ETHUSDT"]], p2.quote_volume[["BTCUSDT", "ETHUSDT"]],
                p2.funding[["BTCUSDT", "ETHUSDT"]], p2.close[["BTCUSDT", "ETHUSDT"]].notna())
    f2 = ml_meta.features(pm2, p2)
    for k in f1:
        pd.testing.assert_frame_equal(f1[k].loc[:t], f2[k].loc[:t], check_names=False)
        assert f1[k].loc[:t].notna().any().any(), k
    # the label of day d uses opens d+1..d+6 only: changing opens after t leaves labels up to t-6 intact
    o3 = pm.open.copy()
    o3.loc[o3.index > t] *= np.linspace(0.5, 2.0, int((o3.index > t).sum()))[:, None]
    pm3 = Panel(o3, pm.high, pm.low, pm.close, pm.quote_volume, pm.funding, pm.universe)
    y1, y3 = ml_meta.labels(pm), ml_meta.labels(pm3)
    cut = t - pd.Timedelta(days=ml_meta.HORIZON + 1)
    pd.testing.assert_frame_equal(y1.loc[:cut], y3.loc[:cut])
    assert not y1.loc[cut + pd.Timedelta(days=1):t].equals(y3.loc[cut + pd.Timedelta(days=1):t])
    # purging + embargo: the last training day is 11 days before the test starts
    days = p.close.index
    te0 = days[200]
    keep = ml_meta.train_mask(days, days[0], te0)
    assert days[keep].max() == te0 - pd.Timedelta(days=ml_meta.HORIZON + 1 + ml_meta.EMBARGO_DAYS)


def test_round2_families_use_no_future_data():
    from cqs.strategies import families_ext
    p = synthetic_panel()
    cols = ["BTCUSDT", "ETHUSDT"]
    pm = Panel(p.open[cols], p.high[cols], p.low[cols], p.close[cols], p.quote_volume[cols], p.funding[cols],
               p.close[cols].notna())
    t = p.close.index[300]
    rng = np.random.default_rng(1)
    spot = pm.close * 1.001
    rf = pd.Series(0.03 + rng.normal(0, 0.002, len(pm.close)), index=pm.close.index)
    dvol = pd.DataFrame(60 + rng.normal(0, 8, pm.close.shape), index=pm.close.index, columns=cols)
    pm2 = cut_future(pm, t)
    rf2, dvol2 = rf.copy(), dvol.copy()
    rf2[rf2.index > t] += 0.5
    dvol2[dvol2.index > t] *= 3
    for margin in (0.0, 0.05):
        w1 = families_ext.r2a_carry_rf(pm, spot, rf, margin)
        w2 = families_ext.r2a_carry_rf(pm2, spot, rf2, margin)
        pd.testing.assert_frame_equal(w1.loc[:t], w2.loc[:t])
    for z, ls in ((0.0, True), (1.0, False)):
        w1 = families_ext.r2c_vrp(pm, dvol, z, ls)
        w2 = families_ext.r2c_vrp(pm2, dvol2, z, ls)
        pd.testing.assert_frame_equal(w1.loc[:t], w2.loc[:t])
        assert w1.abs().sum().sum() > 0
    # R2A trades only when funding beats the risk-free rate by the margin
    w_hi = families_ext.r2a_carry_rf(pm, spot, rf + 1.0, 0.0)
    assert (w_hi == 0).all().all()


def test_r2b_rebalances_on_schedule():
    p = synthetic_panel()
    w = families.h3a_funding_xs(p, 7, rebalance_every=7)
    decided = w.notna().any(axis=1)
    forced = w.loc[~decided.index.isin(decided.index[::7])].notna().any(axis=1)
    # between scheduled rebalances only universe-forced exits (weight 0) may appear
    off = w.loc[forced[forced].index]
    assert ((off.fillna(0) == 0).all(axis=1)).all()
    assert decided.iloc[::7].all()
