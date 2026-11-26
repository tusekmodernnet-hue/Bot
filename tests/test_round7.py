"""Round 7 building blocks on cases with known answers, look-ahead mutation tests and the runner end to end on
synthetic data (R7A multi-coin carry, R7B residual reversion, R7C new-token shorts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest import engine
from cqs.backtest.costs import CostModel
from cqs.data import spotmap
from cqs.research.panel import Panel
from cqs.strategies import carry_multi, listings, statarb

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


def days(n, start="2022-01-01"):
    return pd.date_range(start, periods=n, freq="1D", tz=UTC)


# ------------------------------------------------------------------ spot mapping
def test_spot_candidates_strip_the_contract_size_prefix():
    assert spotmap.candidates("1000PEPEUSDT") == [("1000PEPEUSDT", 1.0), ("PEPEUSDT", 1000.0)]
    assert spotmap.candidates("1MBABYDOGEUSDT")[1] == ("BABYDOGEUSDT", 1e6)
    assert spotmap.candidates("SOLUSDT") == [("SOLUSDT", 1.0)]
    spot = {"PEPEUSDT", "1000SATSUSDT", "SOLUSDT"}
    assert spotmap.match("1000PEPEUSDT", spot) == ("PEPEUSDT", 1000.0)
    assert spotmap.match("1000SATSUSDT", spot) == ("1000SATSUSDT", 1.0)  # the exact name wins
    assert spotmap.match("FARTCOINUSDT", spot) is None


def test_pair_availability_needs_spot_age_and_agreeing_prices():
    idx = days(60)
    perp = pd.DataFrame({"A": 1000.0, "B": 10.0}, index=idx)
    spot = pd.DataFrame({"A": 1.0, "B": np.where(np.arange(60) < 40, 10.0, 20.0)}, index=idx)
    spot.iloc[:5, 0] = np.nan  # A's spot pair starts on day 5
    av = carry_multi.pair_available(perp, spot, {"A": 1000.0, "B": 1.0})
    assert not av["A"].iloc[33] and av["A"].iloc[34]  # 30 spot days on day 34
    assert av["B"].iloc[39] and not av["B"].iloc[43]  # the price ratio broke on day 40: B is another asset now
    assert not carry_multi.pair_available(perp, spot.drop(columns="B"), {})["B"].any()


# ------------------------------------------------------------------ R7A
def test_funding_known_sums_three_days_of_settled_rates():
    ft = pd.date_range("2021-12-01", "2022-02-01", freq="8h", tz=UTC)
    f = pd.DataFrame({"A": 1e-4}, index=ft)
    idx = days(10)
    out = carry_multi.funding_known(f, idx)
    assert out["A"].iloc[3] == pytest.approx(9e-4 * 365 / 3)  # 9 settlements in 3 days
    g = f.copy()
    g.loc[idx[3] + pd.Timedelta(days=1), "A"] = 0.5  # the settlement at the decision itself is not known yet
    assert carry_multi.funding_known(g, idx)["A"].iloc[3] == pytest.approx(out["A"].iloc[3])
    assert carry_multi.funding_known(g, idx)["A"].iloc[4] > 1.0


def test_r7a_fills_slots_by_excess_never_swaps_and_exits_at_half_margin():
    idx = days(6)
    e = pd.DataFrame({"A": [0.30, 0.30, 0.09, 0.30, 0.30, np.nan],
                      "B": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                      "C": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]}, index=idx)
    can_open = pd.DataFrame(True, index=idx, columns=e.columns)
    can_open.loc[idx[0], "C"] = False  # C cannot be opened on day 0 (e.g. not yet in the top 50)
    can_hold = pd.DataFrame(True, index=idx, columns=e.columns)
    can_hold.loc[idx[4], "B"] = False
    h = carry_multi.held_pairs(e, can_open, can_hold, margin=0.20, slots=2)
    assert h.iloc[0].tolist() == [True, True, False]  # A and B, the best that can be opened
    assert h.iloc[1].tolist() == [True, True, False]  # C is better but a held pair is never swapped
    assert h.iloc[2].tolist() == [False, True, True]  # A fell below M/2 = 0.10: out, C takes the slot
    assert h.iloc[3].tolist() == [False, True, True]
    assert h.iloc[4].tolist() == [True, False, True]  # B fails the holding test, A re-enters
    assert h.iloc[5].tolist() == [False, True, True]  # a NaN excess closes A
    w = carry_multi.weights(h, slots=2)
    assert w.loc[idx[0], "A"] == -0.5 and w.loc[idx[0], "A_SPOT"] == 0.5 and w.loc[idx[0], "C_SPOT"] == 0.0


def test_r7a_hedged_pair_earns_the_funding_minus_costs():
    idx = pd.date_range("2022-01-01", periods=5, freq="1D", tz=UTC)
    px = pd.DataFrame({"A": 100.0, "A_SPOT": 100.0}, index=idx)
    W = pd.DataFrame({"A": [-0.5, -0.5, -0.5, -0.5, -0.5], "A_SPOT": [0.5] * 5}, index=idx)
    fund = pd.DataFrame({"A": 0.001}, index=pd.date_range("2022-01-02 08:00", periods=6, freq="8h", tz=UTC))
    cm = CostModel(default_slippage_bps=10.0, slippage_bps={"A_SPOT": 15.0})
    res = engine.run(px, px, px, px, W, cm, funding=fund)
    entry = 0.5 * (0.0005 + 0.001) + 0.5 * (0.0005 + 0.0015)
    # six settlements on the short: +0.5 * 0.001 each, on equity that barely moves
    assert res.equity.iloc[-1] == pytest.approx(1.0 - entry + 6 * 0.5 * 0.001, rel=1e-3)
    assert res.funding_paid < 0  # the short received funding


def _carry_world(seed=0, n=200, coins=("A", "B", "C", "D", "E")):
    rng = np.random.default_rng(seed)
    idx = days(n, "2021-06-01")
    perp = pd.DataFrame({c: 50 * np.exp(np.cumsum(rng.normal(0, 0.03, n))) for c in coins}, index=idx)
    spot = perp * (1 + rng.normal(0, 0.002, perp.shape))
    ft = pd.date_range(idx[0] - pd.Timedelta(days=5), idx[-1] + pd.Timedelta(days=1), freq="8h", tz=UTC)
    fund = pd.DataFrame({c: 1e-4 + rng.normal(0, 3e-4, len(ft)) for c in coins}, index=ft)
    return idx, perp, spot, fund


def _r7a_from(idx, perp, spot, fund, rf, margin=0.10, slots=2):
    av = carry_multi.pair_available(perp, spot, {})
    ex = carry_multi.funding_known(fund, idx).sub(rf, axis=0)
    return carry_multi.r7a(ex, av, av, margin, slots)


def test_r7a_uses_no_future_data():
    idx, perp, spot, fund = _carry_world()
    rf = pd.Series(0.03, index=idx)
    W, h = _r7a_from(idx, perp, spot, fund, rf)
    assert h.to_numpy().any()
    rng = np.random.default_rng(5)
    for k, cut in enumerate(idx[20::30]):
        later = idx > cut
        p2, s2 = perp.copy(), spot.copy()
        p2[later] = perp[later] * np.exp(rng.normal(0, 0.2, p2[later].shape))
        s2[later] = spot[later] * np.exp(rng.normal(0, 0.2, s2[later].shape))
        f2 = fund.copy()
        late_f = f2.index + pd.Timedelta(minutes=5) > cut + pd.Timedelta(days=1)  # unknown at the cut's decision
        f2[late_f] = rng.normal(0, 5e-3, f2[late_f].shape)
        W2, _ = _r7a_from(idx, p2, s2, f2, rf)
        pd.testing.assert_frame_equal(W[idx <= cut], W2[idx <= cut])


# ------------------------------------------------------------------ R7B
def _ou_world(seed=1, n=300, b=0.8, beta=1.3):
    rng = np.random.default_rng(seed)
    idx = days(n, "2021-01-01")
    btc = pd.Series(rng.normal(0, 0.03, n), index=idx)
    X = np.zeros(n)
    for i in range(1, n):
        X[i] = b * X[i - 1] + rng.normal(0, 0.02)
    eps = np.diff(np.concatenate([[0.0], X]))
    coin = pd.DataFrame({"A": beta * btc.to_numpy() + eps, "B": rng.normal(0, 0.04, n)}, index=idx)
    return idx, coin, btc


def test_r7b_scores_match_a_direct_computation(monkeypatch):
    idx, coin, btc = _ou_world()
    uni = pd.DataFrame(True, index=idx, columns=coin.columns)
    s, beta, valid = statarb.scores(coin, btc, uni)
    i = 150
    y, x = coin["A"].to_numpy()[i - 59:i + 1], btc.to_numpy()[i - 59:i + 1]
    bb, aa = np.polyfit(x, y, 1)
    X = np.cumsum(y - aa - bb * x)
    b, a = np.polyfit(X[:-1], X[1:], 1)
    zeta = X[1:] - a - b * X[:-1]
    expect = (X[-1] - a / (1 - b)) / np.sqrt(zeta.var(ddof=1) / (1 - b * b))
    assert valid["A"].iloc[i] and beta["A"].iloc[i] == pytest.approx(bb)
    assert s["A"].iloc[i] == pytest.approx(expect)
    assert abs(X[-1]) < 1e-12  # OLS residuals sum to zero: the score is -m / sigma_eq
    assert np.isnan(s["A"].iloc[58]) and not valid["A"].iloc[58]  # 59 returns are not a full window
    # the mean-reversion filter: with the true b = 0.8, a cap below it rejects most windows
    monkeypatch.setattr(statarb, "B_MAX", 0.5)
    _, _, v_slow = statarb.scores(coin, btc, uni)
    assert v_slow["A"].iloc[59:].mean() < 0.2 < valid["A"].iloc[59:].mean()


def test_r7b_positions_follow_the_paper_thresholds():
    idx = days(9)
    s = pd.DataFrame({"A": [-1.0, -1.3, -0.8, -0.4, 1.3, 1.0, 0.7, 2.0, 2.0],
                      "B": [0.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, np.nan]}, index=idx)
    v = pd.DataFrame(True, index=idx, columns=s.columns)
    u = pd.DataFrame(True, index=idx, columns=s.columns)
    u.loc[idx[6], "B"] = False
    sign = statarb.positions(s, v, u, s_in=1.25)
    assert sign["A"].tolist() == [0, 1, 1, 0, -1, -1, 0, -1, -1]
    assert sign["B"].tolist() == [0, -1, -1, -1, -1, -1, 0, -1, 0]  # out of the universe on day 6, NaN on day 8
    one = statarb.positions(s, v, u, s_in=1.25, max_pos=1)
    assert one.loc[idx[1]].tolist() == [0, -1]  # |1.6| beats |-1.3| for the only slot
    flip = pd.DataFrame({"A": [-1.5, 1.5]}, index=idx[:2])
    assert statarb.positions(flip, v[["A"]].iloc[:2], u[["A"]].iloc[:2], 1.25)["A"].tolist() == [1, -1]


def test_r7b_weights_hedge_beta_in_btc():
    idx = days(2)
    sign = pd.DataFrame({"A": [1.0, 1.0], "B": [-1.0, 0.0]}, index=idx)
    beta = pd.DataFrame({"A": [1.5, 4.0], "B": [0.5, 0.5]}, index=idx)
    w = statarb.weights(sign, beta)
    assert w.loc[idx[0]].tolist() == pytest.approx([0.1, -0.1, -0.15 + 0.05])
    assert w.loc[idx[1], "BTCUSDT"] == pytest.approx(-0.3)  # beta clipped to 3


def test_r7b_uses_no_future_data():
    idx, coin, btc = _ou_world(seed=4)
    coin["C"] = 0.8 * btc + np.random.default_rng(9).normal(0, 0.03, len(idx))
    uni = pd.DataFrame(True, index=idx, columns=coin.columns)
    W, sign = statarb.r7b(coin, btc, uni, 1.25)
    assert (sign != 0).to_numpy().any()
    rng = np.random.default_rng(11)
    for cut in idx[70::40]:
        later = idx > cut
        c2, b2 = coin.copy(), btc.copy()
        c2[later] = rng.normal(0, 0.1, c2[later].shape)
        b2[later] = rng.normal(0, 0.1, int(later.sum()))
        W2, _ = statarb.r7b(c2, b2, uni, 1.25)
        pd.testing.assert_frame_equal(W[idx <= cut], W2[idx <= cut])


# ------------------------------------------------------------------ R7C
def test_new_token_events_apply_the_spot_age_rule_and_exclusions():
    first_perp = {"NEWUSDT": ts("2024-03-01"), "OLDUSDT": ts("2024-03-01"), "LATEUSDT": ts("2024-03-01"),
                  "NOSPOTUSDT": ts("2024-03-01"), "POLUSDT": ts("2024-09-13"), "DEFIUSDT": ts("2021-01-01"),
                  "EARLYUSDT": ts("2020-01-01"), "FUTUREUSDT": ts("2025-06-02"), "BTCUSDT": ts("2020-01-01")}
    first_spot = {"NEWUSDT": ts("2024-02-15"), "OLDUSDT": ts("2024-01-20"), "LATEUSDT": ts("2024-05-01"),
                  "NOSPOTUSDT": None, "POLUSDT": ts("2024-09-13")}
    ev = listings.new_token_events(first_perp, first_spot, ts("2020-01-01"), ts("2025-05-31"))
    assert set(ev.index) == {"NEWUSDT", "LATEUSDT", "NOSPOTUSDT"}  # OLD's spot traded 41 days before D0


def _listing_world(n=150):
    idx = days(n, "2024-01-01")
    close = pd.DataFrame({"BTCUSDT": 40000.0, "X": 1.0, "Y": 2.0, "Z": 3.0}, index=idx)
    close.loc[idx < ts("2024-01-11"), ["X", "Y"]] = np.nan
    close.loc[idx < ts("2024-01-11"), "Z"] = np.nan
    close.loc[idx > ts("2024-02-20"), "Y"] = np.nan  # Y is delisted
    qv = pd.DataFrame({"BTCUSDT": 1e10, "X": 30e6, "Y": 50e6, "Z": 10e6}, index=idx).where(close.notna())
    ev = pd.Series({"X": ts("2024-01-11"), "Y": ts("2024-01-11"), "Z": ts("2024-01-11")}, dtype=object)
    return idx, close, qv, ev


def test_r7c_enters_after_the_delay_holds_h_days_and_respects_liquidity():
    idx, close, qv, ev = _listing_world()
    h = listings.shorts(close, qv, ev, delay=7, hold=30)
    x = h["X"]
    first = x.idxmax()
    assert first == ts("2024-01-18")  # decided at the close of D0 + 7
    assert int(x.sum()) == 30 and x[first:].iloc[:30].all() and not x[first:].iloc[30]
    assert h["Y"].any() and not h.loc[ts("2024-02-21"):, "Y"].any()  # delisted: out when the bar is missing
    assert not h["Z"].any()  # 10M a day is below the 20M liquidity filter
    one = listings.shorts(close, qv, ev, delay=7, hold=30, max_shorts=1)
    assert one["Y"].any() and not one["X"].any()  # the larger volume takes the only slot
    w = listings.weights(h)
    assert w.loc[ts("2024-01-20")].to_dict() == pytest.approx({"X": -0.1, "Y": -0.1, "Z": 0.0, "BTCUSDT": 0.2})


def test_r7c_uses_no_future_data():
    idx, close, qv, ev = _listing_world()
    rng = np.random.default_rng(3)
    base = listings.weights(listings.shorts(close, qv, ev, delay=1, hold=30))
    for cut in idx[5::15]:
        later = idx > cut
        c2, q2 = close.copy(), qv.copy()
        c2[later] = c2[later] * np.exp(rng.normal(0, 0.3, c2[later].shape))
        q2[later] = rng.uniform(0, 1e8, q2[later].shape)
        c2.loc[later & (rng.uniform(size=len(idx)) < 0.05), "X"] = np.nan
        w2 = listings.weights(listings.shorts(c2, q2, ev, delay=1, hold=30))
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
    from cqs.research import run_round7 as R7
    monkeypatch.setattr(metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    rng = np.random.default_rng(0)
    idx = days(1000, "2020-01-01")
    coins = ["BTCUSDT", "ETHUSDT", "AUSDT", "BUSDT"]
    close = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))) for c in coins}, index=idx)
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=1), freq="8h", tz=UTC)
    fund = pd.DataFrame({c: 2e-4 + rng.normal(0, 3e-4, len(ft)) for c in coins}, index=ft)
    rf = pd.Series(0.02, index=idx)
    p = _panel(close, fund)
    # R7A
    spot = close * (1 + rng.normal(0, 0.001, close.shape))
    so = spot.shift(1).fillna(spot)
    bars = tuple(pd.concat([getattr(p, f), s.add_suffix("_SPOT")], axis=1)
                 for f, s in (("open", so), ("high", np.maximum(so, spot)), ("low", np.minimum(so, spot)), ("close", spot)))
    av = carry_multi.pair_available(close, spot, {})
    ex = carry_multi.funding_known(fund, idx).sub(rf, axis=0)
    d7a = dict(p=p, bars=bars, cols=coins, excess=ex, can_open=av, can_hold=av, spot_cols=[c + "_SPOT" for c in coins],
               rf=rf, costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic", mult={})
    # R7B
    lr = np.log(close).diff()
    uni = pd.DataFrame(True, index=idx, columns=coins[1:])
    d7b = dict(p=p, bars=(p.open, p.high, p.low, p.close), logret=lr[coins[1:]], btc=lr["BTCUSDT"], uni=uni, rf=rf,
               costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic")
    # R7C: new contracts every 20 days from 2021-01
    lst = {}
    for k, d0 in enumerate(pd.date_range("2021-01-05", "2022-08-01", freq="20D", tz=UTC)):
        c = pd.Series(np.nan, index=idx)
        live = idx >= d0
        c[live] = 5 * np.exp(np.cumsum(rng.normal(-0.003, 0.05, int(live.sum()))))
        lst[f"N{k}USDT"] = (d0, c)
    cl = pd.concat([close[["BTCUSDT"]], pd.DataFrame({s: v[1] for s, v in lst.items()})], axis=1)
    f7c = pd.DataFrame({s: rng.normal(0, 3e-4, len(ft)) for s in cl.columns}, index=ft)
    p7c = _panel(cl, f7c, qv=cl * 0 + 5e7)
    ev = pd.Series({s: v[0] for s, v in lst.items()}, dtype=object)
    d7c = dict(p=p7c, bars=(p7c.open, p7c.high, p7c.low, p7c.close), events=ev, rf=rf,
               costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic", missing_funding=[])
    return R7, {"R7A": d7a, "R7B": d7b, "R7C": d7c}


def test_every_round7_family_runs_end_to_end(synthetic):
    R7, data = synthetic
    for fam, d in data.items():
        out, oos = R7.run_family(fam, d, log=False)
        names = {R7._name(R7.GRIDS[fam][0], p) for p in R7.GRIDS[fam][1]}
        assert set(out["variants"]) == names, fam
        for name, pv in out["variants"].items():
            assert "ruined" not in pv, (fam, name)
            assert pv["n_trades"] > 0, (fam, name)
        assert set(out["decisions"]) == names
        assert out["walk_forward"]["oos"]["start"] == "2022-01-01", fam
        assert len(oos) > 100


def test_main_logs_ten_trials_writes_the_summary_and_never_reruns(synthetic, monkeypatch, tmp_path):
    from cqs import config
    from cqs.research import registry
    R7, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R7, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R7, "preregistered", lambda: True)
    monkeypatch.setattr(R7, "LOADERS", {f: (lambda f=f: data[f], R7.LOADERS[f][1]) for f in R7.LOADERS})
    assert R7.main([]) == 0
    for f in ("R7A.json", "R7B.json", "R7C.json", "R7A_oos.csv", "R7_summary.json"):
        assert (tmp_path / "results" / f).exists(), f
    ex = [r for r in registry.read() if r.get("type") == "experiment"]
    assert sorted(r["family"] for r in ex) == ["R7A"] * 4 + ["R7B"] * 2 + ["R7C"] * 4
    import json
    summ = json.loads((tmp_path / "results" / "R7_summary.json").read_text())
    assert summ["n_trials_program"] == 10 and set(summ["families"]) == {"R7A", "R7B", "R7C"}
    assert R7.main([]) == 0
    assert len([r for r in registry.read() if r.get("type") == "experiment"]) == 10


def test_a_regime_without_positions_counts_as_sharpe_zero():
    from cqs.research import run_round7 as R7
    reg = {"bull": {"days": 10, "sharpe": float("nan"), "total_return": 0.0},
           "bear": {"days": 10, "sharpe": float("nan"), "total_return": 0.01},
           "range": {"days": 10, "sharpe": -0.5, "total_return": -0.01}}
    out = R7._regime_ok(reg)
    assert out["bull"]["sharpe"] == 0.0 and out["bull"]["no_position"]
    assert np.isnan(out["bear"]["sharpe"]) and out["range"]["sharpe"] == -0.5
