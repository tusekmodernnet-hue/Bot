"""Round 10 building blocks on cases with known answers, a look-ahead mutation test and the runner end to end on
synthetic data (R10A: short liquid tokens after an extreme one-day rise)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest.costs import CostModel
from cqs.research.panel import Panel
from cqs.strategies import listing_risk, pumps

UTC = "UTC"


def days(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="1D", tz=UTC)


def test_pump_events_need_the_rise_and_membership_at_the_previous_close():
    idx = days(6)
    close = pd.DataFrame({"A": [10.0, 10, 13.5, 13.5, 13.5, 20.0],     # +35% on day 2, +48% on day 5
                          "B": [10.0, 10, 14.0, 14.0, 14.0, 14.0]}, index=idx)
    uni = pd.DataFrame(True, index=idx, columns=["A", "B"])
    uni.loc[idx[:2], "B"] = False                                       # B joins only at the close of day 2
    ev = pumps.pump_events(close, uni, 0.3)
    assert ev.to_dict("records") == [{"contract": "A", "d0": idx[2]}, {"contract": "A", "d0": idx[5]}]
    assert len(pumps.pump_events(close, uni, 0.5)) == 0


def _bars(close):
    opn = close.shift(1).fillna(close).where(close.notna())
    return opn, close, close * 0 + 5e7


def test_repeated_events_skip_a_held_contract_and_respect_the_cooldown_after_a_stop():
    idx = days(40)
    c = np.full(40, 10.0)
    c[10:] = 25.0                               # the short entered at the open of day 2 (10.0) is stopped on day 10
    close = pd.DataFrame({"A": c}, index=idx)
    opn, close, qv = _bars(close)
    ev = pd.DataFrame({"contract": ["A", "A", "A", "A"], "d0": [idx[0], idx[3], idx[12], idx[25]]})
    held, stops = pumps.event_shorts(opn, close, qv, ev, 1, 10, stop_k=1.0)
    a = held["A"]
    assert a.iloc[1:10].all() and not a.iloc[10]          # first event; the one of day 3 is skipped (A is held)
    assert len(stops) == 1 and stops[0]["stopped"] == idx[10]
    assert not a.iloc[13]                                  # decided at day 13, inside the 10-day cooldown
    assert a.iloc[26:36].all() and not a.iloc[36:].any()   # day 26 is after the cooldown: a new 10-day short


def test_single_events_give_exactly_the_r9a_shorts():
    rng = np.random.default_rng(4)
    idx = days(200)
    close = pd.DataFrame({f"N{j}": 10 * np.exp(np.cumsum(rng.normal(0, 0.1, 200))) for j in range(8)}, index=idx)
    for j in range(8):
        close.iloc[:5 + 20 * j, j] = np.nan
    opn, close, qv = _bars(close)
    qv = qv * rng.uniform(0.2, 2.0, qv.shape)
    d0 = {f"N{j}": idx[5 + 20 * j] for j in range(8)}
    ev = pd.DataFrame({"contract": list(d0), "d0": list(d0.values())})
    for k in (None, 0.5):
        a, sa = pumps.event_shorts(opn, close, qv, ev, 1, 30, max_shorts=3, stop_k=k)
        b, sb = listing_risk.shorts_with_stop(opn, close, qv, pd.Series(d0, dtype=object), 1, 30, max_shorts=3,
                                              stop_k=k)
        pd.testing.assert_frame_equal(a, b.reindex(columns=a.columns))
        assert sa == sb


def test_r10a_uses_no_future_data():
    rng = np.random.default_rng(6)
    idx = days(300)
    names = [f"S{i}" for i in range(10)]
    close = pd.DataFrame({s: 10 * np.exp(np.cumsum(rng.normal(0, 0.12, 300))) for s in names}, index=idx)
    uni = pd.DataFrame(True, index=idx, columns=names)
    basket = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])
    def weights(cl):
        cl = cl.assign(A=1.0, B=1.0, C=1.0)
        opn, cl, qv = _bars(cl)
        ev = pumps.pump_events(cl[names], uni, 0.2)
        return pumps.r10a(opn, cl, qv, ev, basket, 10)[0]
    base = weights(close)
    assert (base < 0).any().sum() >= 3
    for cut in idx[20::40]:
        later = idx > cut
        c2 = close.copy()
        c2[later] = c2[later] * np.exp(rng.normal(0, 0.4, c2[later].shape))
        pd.testing.assert_frame_equal(base[idx <= cut], weights(c2)[idx <= cut])


# ------------------------------------------------------------------ runner end to end
def _panel(close: pd.DataFrame, funding: pd.DataFrame) -> Panel:
    o = close.shift(1).fillna(close).where(close.notna())
    return Panel(o, np.maximum(o, close) * 1.001, np.minimum(o, close) * 0.999, close, close * 0 + 5e7, funding,
                 pd.DataFrame(True, index=close.index, columns=close.columns))


@pytest.fixture
def synthetic(monkeypatch):
    from cqs.research import metrics_window
    from cqs.research import run_round10 as R10
    monkeypatch.setattr(metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    rng = np.random.default_rng(3)
    idx = days(1000, "2020-01-01")
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=1), freq="8h", tz=UTC)
    alts = [f"A{i}USDT" for i in range(6)]
    tok = [f"T{i}USDT" for i in range(12)]
    close = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))) for c in ["BTCUSDT"] + alts},
                         index=idx)
    for t in tok:
        r = rng.normal(-0.001, 0.06, len(idx))
        r[rng.uniform(size=len(idx)) < 0.01] = 0.6           # occasional one-day jumps above both thresholds
        close[t] = 5 * np.exp(np.cumsum(np.log1p(r)))
    fund = pd.DataFrame({s: rng.normal(0, 3e-4, len(ft)) for s in close.columns}, index=ft)
    p = _panel(close, fund)
    basket = pd.DataFrame(False, index=idx, columns=close.columns)
    basket[alts] = True
    uni = pd.DataFrame(True, index=idx, columns=tok)
    events = {x: pumps.pump_events(close[tok], uni, x) for x in R10.THRESHOLDS}
    d = dict(p=p, bars=(p.open, p.high, p.low, p.close), events=pd.Series(index=tok, dtype=object), basket=basket,
             rf=pd.Series(0.02, index=idx), costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic",
             missing_funding=[], pump_events=events,
             events_count={str(x): int(len(e)) for x, e in events.items()})
    return R10, {"R10A": d}


def test_round10_runs_end_to_end(synthetic):
    R10, data = synthetic
    out, oos = R10.run_family("R10A", data["R10A"], log=False)
    names = {R10._name(R10.GRIDS["R10A"][0], p) for p in R10.GRIDS["R10A"][1]}
    assert set(out["variants"]) == names and set(out["decisions"]) == names
    for name, pv in out["variants"].items():
        assert "ruined" not in pv and pv["n_trades"] > 0, name
        assert out["decisions"][name]["shorts_opened"] > 0
    assert out["walk_forward"]["oos"]["start"] == "2022-01-01" and len(oos) > 100


def test_round10_main_logs_four_trials_and_never_reruns(synthetic, monkeypatch, tmp_path):
    import json

    from cqs import config
    from cqs.research import registry
    R10, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R10, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R10, "preregistered", lambda: True)
    monkeypatch.setattr(R10, "LOADERS", {f: (lambda f=f: data[f], R10.LOADERS[f][1]) for f in R10.LOADERS})
    assert R10.main([]) == 0
    for f in ("R10A.json", "R10A_oos.csv", "R10_summary.json"):
        assert (tmp_path / "results" / f).exists(), f
    ex = [r for r in registry.read() if r.get("type") == "experiment"]
    assert sorted(r["family"] for r in ex) == ["R10A"] * 4
    summ = json.loads((tmp_path / "results" / "R10_summary.json").read_text())
    assert summ["n_trials_program"] == 4
    assert R10.main([]) == 0
    assert len([r for r in registry.read() if r.get("type") == "experiment"]) == 4


def test_round10_refuses_contracts_without_funding_data(synthetic, monkeypatch, tmp_path):
    from cqs import config
    R10, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R10, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R10, "preregistered", lambda: True)
    bad = data["R10A"] | {"missing_funding": ["T1USDT"]}
    monkeypatch.setattr(R10, "LOADERS", {"R10A": (lambda: bad, R10.LOADERS["R10A"][1])})
    with pytest.raises(SystemExit, match="no funding data"):
        R10.main(["R10A"])
