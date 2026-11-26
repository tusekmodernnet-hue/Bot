"""Round 11 building blocks on cases with known answers, a look-ahead mutation test and the runner end to end on
synthetic data (R11A: short older liquid tokens after a day of extremely negative funding)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest.costs import CostModel
from cqs.research.panel import Panel
from cqs.strategies import pumps, short_demand

UTC = "UTC"


def days(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="1D", tz=UTC)


def test_daily_funding_sums_one_utc_day_whatever_the_interval():
    idx = days(3)
    t0 = idx[0]
    a = pd.Series(-0.001, index=pd.date_range(t0, periods=7, freq="8h"))              # 8h: 3 a day, + 00:00 of day 2
    b = pd.Series(-0.0005, index=pd.date_range(t0, periods=12, freq="4h"))            # 4h: 6 a day, 2 days
    c = pd.Series(-0.0001, index=pd.date_range(t0 + pd.Timedelta(milliseconds=3), periods=24, freq="1h"))
    f = pd.DataFrame({"A": a, "B": b, "C": c}).sort_index()
    d = short_demand.daily_funding(f, idx)
    assert d.loc[idx[0], "A"] == pytest.approx(-0.003) and d.loc[idx[1], "A"] == pytest.approx(-0.003)
    assert d.loc[idx[2], "A"] == pytest.approx(-0.001)                                # the 00:00 print of day 2
    assert d.loc[idx[0], "B"] == pytest.approx(-0.003) and d.loc[idx[1], "B"] == pytest.approx(-0.003)
    assert np.isnan(d.loc[idx[2], "B"])
    assert d.loc[idx[0], "C"] == pytest.approx(-0.0024) and np.isnan(d.loc[idx[1], "C"])


def test_funding_events_need_the_threshold_and_membership_at_the_previous_close():
    idx = days(5)
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=16), freq="8h")
    f = pd.DataFrame({"A": 0.0001, "B": 0.0001}, index=ft)
    day = f.index.floor("1D")
    f.loc[day == idx[2], "A"] = -0.0011       # day 2: -0.33%
    f.loc[day == idx[3], "A"] = -0.0025       # day 3: -0.75%
    f.loc[day == idx[1], "B"] = -0.0025       # day 1: -0.75%
    uni = pd.DataFrame(True, index=idx, columns=["A", "B"])

    def rec(ev):
        return [(r["contract"], r["d0"]) for r in ev.to_dict("records")]
    assert rec(short_demand.funding_events(f, uni, 0.003, idx)) == [("B", idx[1]), ("A", idx[2]), ("A", idx[3])]
    assert rec(short_demand.funding_events(f, uni, 0.006, idx)) == [("B", idx[1]), ("A", idx[3])]
    late = uni.copy()
    late.loc[idx[0], "B"] = False             # B joins the universe only at the close of day 1
    assert rec(short_demand.funding_events(f, late, 0.003, idx)) == [("A", idx[2]), ("A", idx[3])]


def test_load_funding_reports_contracts_without_data(monkeypatch):
    def fake(s):
        if s == "X":
            raise FileNotFoundError(s)
        if s == "Y":
            return pd.DataFrame({"funding_time": pd.DatetimeIndex([], tz=UTC), "rate": []})
        return pd.DataFrame({"funding_time": days(2), "rate": [0.0001, -0.0002]})
    monkeypatch.setattr(short_demand.store, "load_funding", fake)
    f, missing = short_demand.load_funding(["W", "X", "Y"])
    assert list(f.columns) == ["W"] and missing == ["X", "Y"]


def _bars(close):
    opn = close.shift(1).fillna(close).where(close.notna())
    return opn, close, close * 0 + 5e7


def test_r11a_shorts_the_day_after_the_event_and_hedges_with_the_basket():
    idx = days(30)
    close = pd.DataFrame({"S": 10.0, "A": 20.0, "B": 30.0}, index=idx)
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=16), freq="8h")
    f = pd.DataFrame(0.0001, index=ft, columns=["S", "A", "B"])
    f.loc[f.index.floor("1D") == idx[5], "S"] = -0.002
    uni = pd.DataFrame(True, index=idx, columns=["S"])
    ev = short_demand.funding_events(f, uni, 0.003, idx)
    basket = pd.DataFrame(False, index=idx, columns=["S", "A", "B"])
    basket[["A", "B"]] = True
    opn, close, qv = _bars(close)
    W, held, stops = short_demand.r11a(opn, close, qv, ev, basket, 10)
    s = held["S"]
    assert not s.iloc[:6].any() and s.iloc[6:16].all() and not s.iloc[16:].any()   # decided at the close of day 6
    assert (W.loc[s, "S"] == -pumps.SIZE).all() and np.allclose(W.loc[s, ["A", "B"]], pumps.SIZE / 2)
    assert (W.loc[~s] == 0).all().all() and stops == []


def test_r11a_uses_no_future_data():
    rng = np.random.default_rng(7)
    idx = days(300)
    names = [f"S{i}" for i in range(10)]
    close = pd.DataFrame({s: 10 * np.exp(np.cumsum(rng.normal(0, 0.08, 300))) for s in names}, index=idx)
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=16), freq="8h")
    fund = pd.DataFrame(rng.normal(-0.0004, 0.0012, (len(ft), len(names))), index=ft, columns=names)
    uni = pd.DataFrame(True, index=idx, columns=names)
    basket = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])

    def weights(cl, fu):
        cl = cl.assign(A=1.0, B=1.0, C=1.0)
        opn, cl, qv = _bars(cl)
        ev = short_demand.funding_events(fu, uni, 0.003, idx)
        return short_demand.r11a(opn, cl, qv, ev, basket, 10)[0]
    base = weights(close, fund)
    assert (base < 0).any().sum() >= 3
    for cut in idx[20::40]:
        later, flater = idx > cut, fund.index >= cut
        c2, f2 = close.copy(), fund.copy()
        c2[later] = c2[later] * np.exp(rng.normal(0, 0.4, c2[later].shape))
        f2[flater] = rng.normal(-0.003, 0.003, f2[flater].shape)
        pd.testing.assert_frame_equal(base[idx <= cut], weights(c2, f2)[idx <= cut])


# ------------------------------------------------------------------ runner end to end
def _panel(close: pd.DataFrame, funding: pd.DataFrame) -> Panel:
    o = close.shift(1).fillna(close).where(close.notna())
    return Panel(o, np.maximum(o, close) * 1.001, np.minimum(o, close) * 0.999, close, close * 0 + 5e7, funding,
                 pd.DataFrame(True, index=close.index, columns=close.columns))


@pytest.fixture
def synthetic(monkeypatch):
    from cqs.research import metrics_window
    from cqs.research import run_round11 as R11
    monkeypatch.setattr(metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    rng = np.random.default_rng(5)
    idx = days(1000, "2020-01-01")
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=1), freq="8h", tz=UTC)
    alts = [f"A{i}USDT" for i in range(6)]
    tok = [f"T{i}USDT" for i in range(12)]
    close = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))) for c in ["BTCUSDT"] + alts},
                         index=idx)
    for t in tok:
        close[t] = 5 * np.exp(np.cumsum(rng.normal(-0.0005, 0.05, len(idx))))
    fund = pd.DataFrame({s: rng.normal(0, 3e-4, len(ft)) for s in close.columns}, index=ft)
    for t in tok:                                            # occasional days below both thresholds
        bad = rng.uniform(size=len(idx)) < 0.02
        fund.loc[np.isin(fund.index.floor("1D"), idx[bad]), t] = -0.0025
    p = _panel(close, fund)
    basket = pd.DataFrame(False, index=idx, columns=close.columns)
    basket[alts] = True
    uni = pd.DataFrame(True, index=idx, columns=tok)
    events = {x: short_demand.funding_events(fund[tok], uni, x, idx) for x in R11.THRESHOLDS}
    d = dict(p=p, bars=(p.open, p.high, p.low, p.close), events=pd.Series(index=tok, dtype=object), basket=basket,
             rf=pd.Series(0.02, index=idx), costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic",
             missing_funding=[], pump_events=events, universe_contracts=len(tok),
             events_count={str(x): int(len(e)) for x, e in events.items()})
    return R11, {"R11A": d}


def test_round11_runs_end_to_end(synthetic):
    R11, data = synthetic
    out, oos = R11.run_family("R11A", data["R11A"], log=False)
    names = {R11._name(R11.GRIDS["R11A"][0], p) for p in R11.GRIDS["R11A"][1]}
    assert set(out["variants"]) == names and set(out["decisions"]) == names
    for name, pv in out["variants"].items():
        assert "ruined" not in pv and pv["n_trades"] > 0, name
        dec = out["decisions"][name]
        assert dec["shorts_opened"] > 0 and dec["mean_daily_funding_of_held_shorts"] < 0, name
    assert out["walk_forward"]["oos"]["start"] == "2022-01-01" and len(oos) > 100
    assert out["universe_contracts"] == 12


def test_round11_main_logs_four_trials_and_never_reruns(synthetic, monkeypatch, tmp_path):
    import json

    from cqs import config
    from cqs.research import registry
    R11, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R11, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R11, "preregistered", lambda: True)
    monkeypatch.setattr(R11, "LOADERS", {f: (lambda f=f: data[f], R11.LOADERS[f][1]) for f in R11.LOADERS})
    assert R11.main([]) == 0
    for f in ("R11A.json", "R11A_oos.csv", "R11_summary.json"):
        assert (tmp_path / "results" / f).exists(), f
    ex = [r for r in registry.read() if r.get("type") == "experiment"]
    assert sorted(r["family"] for r in ex) == ["R11A"] * 4
    summ = json.loads((tmp_path / "results" / "R11_summary.json").read_text())
    assert summ["n_trials_program"] == 4
    assert R11.main([]) == 0
    assert len([r for r in registry.read() if r.get("type") == "experiment"]) == 4


def test_round11_refuses_contracts_without_funding_data(synthetic, monkeypatch, tmp_path):
    from cqs import config
    R11, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R11, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R11, "preregistered", lambda: True)
    bad = data["R11A"] | {"missing_funding": ["T1USDT"]}
    monkeypatch.setattr(R11, "LOADERS", {"R11A": (lambda: bad, R11.LOADERS["R11A"][1])})
    with pytest.raises(SystemExit, match="no funding data"):
        R11.main(["R11A"])
