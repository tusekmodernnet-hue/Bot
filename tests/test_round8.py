"""Round 8 building blocks on cases with known answers, look-ahead mutation tests and the runner end to end on
synthetic data (R8A new-token shorts with an alt basket, R8B new perpetuals on old tokens, R8C token age)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest import engine
from cqs.backtest.costs import CostModel
from cqs.research.panel import Panel
from cqs.strategies import listings, supply

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


def days(n, start="2022-01-01"):
    return pd.date_range(start, periods=n, freq="1D", tz=UTC)


# ------------------------------------------------------------------ engine: relative rebalance band
def _bars(close: pd.DataFrame):
    o = close.shift(1).fillna(close)
    return o, np.maximum(o, close), np.minimum(o, close), close


def test_relative_band_opens_small_positions_and_ignores_small_drift():
    idx = days(8)
    close = pd.DataFrame({"A": [100.0, 100, 100, 110, 110, 130, 130, 130], "B": 100.0}, index=idx)
    w = pd.DataFrame({"A": 0.005, "B": 0.0}, index=idx)
    w.iloc[-2:, 0] = 0.0
    free = CostModel(taker_fee=0.0, default_slippage_bps=0.0)
    # an absolute band of 2% of equity never opens a 0.5% position
    stuck = engine.run(*_bars(close), w, free, rebalance_band=0.02)
    assert (stuck.positions["A"] == 0).all()
    res = engine.run(*_bars(close), w, free, rebalance_band_rel=0.2)
    pos = res.positions["A"]
    assert pos.iloc[1] == pytest.approx(0.005)                       # opened at the next open
    assert pos.iloc[3] == pytest.approx(0.005 * 1.10)                # +10% drift: inside the 20% band, kept
    assert pos.iloc[5] == pytest.approx(0.005 * 1.10 * 130 / 110)    # day 5 opens at 110: drift still inside
    assert pos.iloc[6] == pytest.approx(0.005 * 1.0015, rel=1e-9)    # +30% vs target: trimmed to 0.5% of equity
    assert pos.iloc[7] == 0.0                                        # the exit to 0 always trades


def test_relative_band_off_keeps_the_old_behaviour():
    idx = days(6)
    close = pd.DataFrame({"A": [100.0, 101, 99, 102, 98, 100]}, index=idx)
    w = pd.DataFrame({"A": 0.5}, index=idx)
    c = CostModel()
    a = engine.run(*_bars(close), w, c, rebalance_band=0.02)
    b = engine.run(*_bars(close), w, c, rebalance_band=0.02, rebalance_band_rel=0.0)
    pd.testing.assert_series_equal(a.equity, b.equity)


# ------------------------------------------------------------------ R8B events
def test_listing_consistency_uses_the_multiplier_and_only_the_first_two_days():
    idx = days(10, "2023-05-01")
    spot = pd.Series(0.001, index=idx)
    perp = pd.Series(1.0, index=idx)  # a 1000x contract
    assert supply.listing_consistent(perp, spot, 1000.0, ts("2023-05-03"))
    assert not supply.listing_consistent(perp * 1.5, spot, 1000.0, ts("2023-05-03"))
    later = perp.copy()
    later[idx > ts("2023-05-04")] = 9.0  # after D0 + 1: ignored
    assert supply.listing_consistent(later, spot, 1000.0, ts("2023-05-03"))
    assert not supply.listing_consistent(perp, spot.where(idx < ts("2023-05-02")), 1000.0, ts("2023-05-03"))


def test_old_token_events_are_the_listings_r7c_leaves_out():
    first_perp = {"OLDUSDT": ts("2024-03-01"), "NEWUSDT": ts("2024-03-01"), "NOSPOTUSDT": ts("2024-03-01"),
                  "BADUSDT": ts("2024-03-01"), "POLUSDT": ts("2024-09-13"), "EARLYUSDT": ts("2020-01-01"),
                  "LATEUSDT": ts("2025-06-02"), "BTCUSDT": ts("2020-01-01")}
    first_spot = {"OLDUSDT": ts("2023-01-20"), "NEWUSDT": ts("2024-02-15"), "NOSPOTUSDT": None,
                  "BADUSDT": ts("2022-01-01"), "POLUSDT": ts("2020-01-01"), "EARLYUSDT": ts("2018-01-01"),
                  "LATEUSDT": ts("2021-01-01"), "BTCUSDT": ts("2017-08-17")}
    consistent = {k: True for k in first_perp} | {"BADUSDT": False}
    ev = supply.old_token_events(first_perp, first_spot, consistent, ts("2020-01-01"), ts("2025-05-31"))
    assert list(ev.index) == ["OLDUSDT"]
    new = listings.new_token_events(first_perp, first_spot, ts("2020-01-01"), ts("2025-05-31"))
    assert not set(ev.index) & set(new.index)


# ------------------------------------------------------------------ basket hedge
def test_basket_hedge_is_dollar_neutral_equal_weight_and_skips_shorted_or_missing_members():
    idx = days(4)
    held = pd.DataFrame({"X": [True, True, True, False], "Y": [False, True, True, False]}, index=idx)
    basket = pd.DataFrame({"A": True, "B": True, "C": True, "Y": True}, index=idx)
    basket.iloc[2, :3] = False  # day 2: only Y is a member, and Y is shorted
    close = pd.DataFrame({"X": 1.0, "Y": 1.0, "A": 1.0, "B": 1.0, "C": 1.0}, index=idx)
    close.loc[idx[1], "C"] = np.nan
    w = supply.basket_weights(held, basket, close)
    # day 0: Y is a member and not shorted yet, so four members share the 0.1 hedge
    assert w.loc[idx[0]].to_dict() == pytest.approx({"X": -0.1, "Y": 0.025, "A": 0.025, "B": 0.025, "C": 0.025})
    # Y is a member but shorted: left out; C has no bar: left out
    assert w.loc[idx[1]].to_dict() == pytest.approx({"X": -0.1, "Y": -0.1, "A": 0.1, "B": 0.1, "C": 0.0})
    assert (w.loc[idx[2]] == 0).all()   # no member left: no unhedged shorts
    assert (w.loc[idx[3]] == 0).all()
    assert w.sum(axis=1).abs().max() < 1e-12


def _event_world(n=200, seed=4):
    rng = np.random.default_rng(seed)
    idx = days(n, "2024-01-01")
    close = pd.DataFrame({c: 10 * np.exp(np.cumsum(rng.normal(0, 0.04, n))) for c in ("A", "B", "C", "X", "Y")},
                         index=idx)
    close.loc[idx < ts("2024-02-01"), ["X", "Y"]] = np.nan
    qv = close * 0 + 5e7
    ev = pd.Series({"X": ts("2024-02-01"), "Y": ts("2024-02-10")}, dtype=object)
    basket = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])
    basket.loc[idx[::7], "C"] = False
    return idx, close, qv, ev, basket


def test_event_shorts_with_the_basket_use_no_future_data():
    idx, close, qv, ev, basket = _event_world()
    rng = np.random.default_rng(5)
    base, _ = supply.r8_event_shorts(close, qv, ev, basket, 1, 30)
    assert (base < 0).any().sum() == 2 and base.sum(axis=1).abs().max() < 1e-12
    for cut in idx[10::25]:
        later = idx > cut
        c2, q2, b2 = close.copy(), qv.copy(), basket.copy()
        c2[later] = c2[later] * np.exp(rng.normal(0, 0.3, c2[later].shape))
        q2[later] = rng.uniform(0, 1e8, q2[later].shape)
        b2[later] = rng.uniform(size=b2[later].shape) < 0.5
        w2, _ = supply.r8_event_shorts(c2, q2, ev, b2, 1, 30)
        pd.testing.assert_frame_equal(base[idx <= cut], w2[idx <= cut])


# ------------------------------------------------------------------ R8C token age
def test_token_birth_is_the_first_binance_listing():
    b = supply.token_birth({"A": ts("2023-01-10"), "B": ts("2023-01-10"), "C": ts("2023-01-10")},
                           {"A": ts("2021-05-01"), "B": None, "C": ts("2023-02-01")})
    assert b == {"A": ts("2021-05-01"), "B": ts("2023-01-10"), "C": ts("2023-01-10")}
    age = supply.age_days(days(3, "2023-01-10"), b)
    assert age["B"].tolist() == [0, 1, 2] and age.loc[ts("2023-01-10"), "A"] == 619


def test_age_membership_changes_only_at_the_sunday_close():
    idx = days(21, "2024-01-01")  # 2024-01-07 is a Sunday
    uni = pd.DataFrame(True, index=idx, columns=["Y", "O", "POLUSDT", "BTCUSDT"])
    age = pd.DataFrame({"Y": 100 + np.arange(21), "O": 800 + np.arange(21), "POLUSDT": 100, "BTCUSDT": 3000},
                       index=idx)
    age.loc[ts("2024-01-09"):, "Y"] = 400  # crosses the 365-day line on a Tuesday
    young, old = supply.age_legs(uni, age, 365)
    assert "POLUSDT" not in young.columns and "BTCUSDT" not in young.columns  # rename and BTC excluded
    assert not young.loc[:ts("2024-01-06")].any().any()   # nothing before the first Sunday close
    assert young.loc[ts("2024-01-07"):ts("2024-01-13"), "Y"].all()  # kept until the next Sunday
    assert not young.loc[ts("2024-01-14"):, "Y"].any()
    assert old.loc[ts("2024-01-07"):, "O"].all()


def test_age_weights_cap_young_names_need_three_per_leg_and_stay_dollar_neutral():
    idx = days(4, "2024-01-01")
    names = [f"Y{i}" for i in range(10)] + [f"O{i}" for i in range(4)]
    young = pd.DataFrame(False, index=idx, columns=names)
    old = young.copy()
    young.iloc[0, :3] = True        # 3 young: -0.1 each
    young.iloc[1, :10] = True       # 10 young: -0.05 each
    young.iloc[2, :2] = True        # 2 young: no position
    young.iloc[3, :3] = True        # 3 young, but one has no bar: no position
    old.loc[:, ["O0", "O1", "O2", "O3"]] = True
    close = pd.DataFrame(1.0, index=idx, columns=names)
    close.loc[idx[3], "Y0"] = np.nan
    w = supply.age_weights(young, old, close)
    assert w.loc[idx[0], ["Y0", "Y1", "Y2"]].tolist() == pytest.approx([-0.1] * 3)
    assert w.loc[idx[0], ["O0", "O1", "O2", "O3"]].tolist() == pytest.approx([0.075] * 4)
    assert w.loc[idx[1], "Y9"] == pytest.approx(-0.05) and w.loc[idx[1], "O0"] == pytest.approx(0.125)
    assert (w.loc[idx[2]] == 0).all() and (w.loc[idx[3]] == 0).all()
    assert w.sum(axis=1).abs().max() < 1e-12


def test_r8c_uses_no_future_data():
    rng = np.random.default_rng(7)
    idx = days(400, "2022-01-01")
    names = [f"S{i}" for i in range(12)]
    close = pd.DataFrame({s: np.exp(np.cumsum(rng.normal(0, 0.05, len(idx)))) for s in names}, index=idx)
    uni = pd.DataFrame(rng.uniform(size=close.shape) < 0.8, index=idx, columns=names)
    birth = {s: ts("2022-01-01") - pd.Timedelta(days=int(d)) for s, d in zip(names, rng.integers(0, 1500, 12))}
    age = supply.age_days(idx, birth)
    base, _ = supply.r8c(uni, age, close, 365)
    assert (base != 0).any().any()
    for cut in idx[20::60]:
        later = idx > cut
        c2, u2 = close.copy(), uni.copy()
        c2[later] = c2[later] * np.exp(rng.normal(0, 0.3, c2[later].shape))
        c2[later & (rng.uniform(size=len(idx)) < 0.1)] = np.nan
        u2[later] = rng.uniform(size=u2[later].shape) < 0.5
        w2, _ = supply.r8c(u2, age, c2, 365)
        pd.testing.assert_frame_equal(base[idx <= cut], w2[idx <= cut])


# ------------------------------------------------------------------ runner end to end
def _panel(close: pd.DataFrame, funding: pd.DataFrame, qv: pd.DataFrame | None = None) -> Panel:
    o = close.shift(1).fillna(close).where(close.notna())
    return Panel(o, np.maximum(o, close) * 1.001, np.minimum(o, close) * 0.999, close,
                 qv if qv is not None else close * 0 + 5e7, funding,
                 pd.DataFrame(True, index=close.index, columns=close.columns))


@pytest.fixture
def synthetic(monkeypatch):
    from cqs.research import metrics_window
    from cqs.research import run_round8 as R8
    monkeypatch.setattr(metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    rng = np.random.default_rng(0)
    idx = days(1000, "2020-01-01")
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=1), freq="8h", tz=UTC)
    rf = pd.Series(0.02, index=idx)
    alts = [f"A{i}USDT" for i in range(6)]
    base = {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))) for c in ["BTCUSDT"] + alts}
    lst = {}
    for k, d0 in enumerate(pd.date_range("2020-03-05", "2022-08-01", freq="20D", tz=UTC)):
        c = pd.Series(np.nan, index=idx)
        live = idx >= d0
        c[live] = 5 * np.exp(np.cumsum(rng.normal(-0.003, 0.05, int(live.sum()))))
        lst[f"N{k}USDT"] = (d0, c)
    close = pd.concat([pd.DataFrame(base, index=idx), pd.DataFrame({s: v[1] for s, v in lst.items()})], axis=1)
    fund = pd.DataFrame({s: rng.normal(0, 3e-4, len(ft)) for s in close.columns}, index=ft)
    p = _panel(close, fund)
    basket = pd.DataFrame(False, index=idx, columns=close.columns)
    basket[alts] = True
    ev = pd.Series({s: v[0] for s, v in lst.items()}, dtype=object)
    common = dict(p=p, bars=(p.open, p.high, p.low, p.close), basket=basket, rf=rf,
                  costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic", missing_funding=[])
    d8a = common | dict(events=ev)
    d8b = common | dict(events=ev.iloc[::2], checked=len(ev), inconsistent=[])
    uni = close.notna() & (close.notna().cumsum() >= 90)
    uni["BTCUSDT"] = False
    birth = {s: v[0] for s, v in lst.items()} | {a: ts("2017-09-01") for a in alts}
    d8c = dict(p=p, bars=(p.open, p.high, p.low, p.close), universe=uni, age=supply.age_days(idx, birth), rf=rf,
               costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic", missing_funding=[])
    return R8, {"R8A": d8a, "R8B": d8b, "R8C": d8c}


def test_every_round8_family_runs_end_to_end(synthetic):
    R8, data = synthetic
    for fam, d in data.items():
        out, oos = R8.run_family(fam, d, log=False)
        names = {R8._name(R8.GRIDS[fam][0], p) for p in R8.GRIDS[fam][1]}
        assert set(out["variants"]) == names, fam
        for name, pv in out["variants"].items():
            assert "ruined" not in pv, (fam, name)
            assert pv["n_trades"] > 0, (fam, name)
        assert set(out["decisions"]) == names
        assert out["walk_forward"]["oos"]["start"] == "2022-01-01", fam
        assert len(oos) > 100


def test_round8_main_logs_ten_trials_writes_the_summary_and_never_reruns(synthetic, monkeypatch, tmp_path):
    from cqs import config
    from cqs.research import registry
    R8, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R8, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R8, "preregistered", lambda: True)
    monkeypatch.setattr(R8, "LOADERS", {f: (lambda f=f: data[f], R8.LOADERS[f][1]) for f in R8.LOADERS})
    assert R8.main([]) == 0
    for f in ("R8A.json", "R8B.json", "R8C.json", "R8C_oos.csv", "R8_summary.json"):
        assert (tmp_path / "results" / f).exists(), f
    ex = [r for r in registry.read() if r.get("type") == "experiment"]
    assert sorted(r["family"] for r in ex) == ["R8A"] * 4 + ["R8B"] * 4 + ["R8C"] * 2
    import json
    summ = json.loads((tmp_path / "results" / "R8_summary.json").read_text())
    assert summ["n_trials_program"] == 10 and set(summ["families"]) == {"R8A", "R8B", "R8C"}
    assert R8.main([]) == 0
    assert len([r for r in registry.read() if r.get("type") == "experiment"]) == 10


def test_round8_refuses_a_family_without_funding_data(synthetic, monkeypatch, tmp_path):
    from cqs import config
    R8, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R8, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R8, "preregistered", lambda: True)
    bad = data["R8C"] | {"missing_funding": ["N1USDT"]}
    monkeypatch.setattr(R8, "LOADERS", {"R8C": (lambda: bad, R8.LOADERS["R8C"][1])})
    with pytest.raises(SystemExit, match="no funding data"):
        R8.main(["R8C"])
