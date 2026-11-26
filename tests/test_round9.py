"""Round 9 building blocks on cases with known answers, a look-ahead mutation test and the runner end to end on
synthetic data (R9A: pooled new-perpetual shorts with a close-based stop and the alt basket)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest.costs import CostModel
from cqs.research.panel import Panel
from cqs.strategies import listing_risk, listings

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


def days(n, start="2022-01-01"):
    return pd.date_range(start, periods=n, freq="1D", tz=UTC)


def _world(n=260, k=14, seed=3):
    rng = np.random.default_rng(seed)
    idx = days(n, "2023-01-01")
    close, ev = {}, {}
    for j in range(k):
        d0 = idx[5 + 15 * j]
        c = pd.Series(np.nan, index=idx)
        live = idx >= d0
        c[live] = 10 * np.exp(np.cumsum(rng.normal(0.0, 0.12, int(live.sum()))))
        close[f"N{j}"], ev[f"N{j}"] = c, d0
    close = pd.DataFrame(close)
    opn = close.shift(1).where(close.notna())
    opn = opn.fillna(close)
    qv = pd.DataFrame(rng.uniform(5e6, 8e7, close.shape), index=idx, columns=close.columns).where(close.notna())
    return idx, opn, close, qv, pd.Series(ev, dtype=object)


# ------------------------------------------------------------------ the stop
def test_without_a_stop_the_shorts_are_exactly_those_of_r7c():
    idx, opn, close, qv, ev = _world()
    for mx in (2, 20):
        held, stops = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30, max_shorts=mx, stop_k=None)
        pd.testing.assert_frame_equal(held, listings.shorts(close, qv, ev, 1, 30, max_shorts=mx))
        assert stops == []


def _stop_case():
    idx = days(12, "2024-03-01")
    close = pd.DataFrame({"X": [9.0, 9.5, 12.0, 14.9, 15.0, 11.0, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                          "Y": [np.nan, np.nan, np.nan, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]}, index=idx)
    opn = close.shift(1).fillna(close).where(close.notna())
    opn.loc[idx[2], "X"] = 10.0          # entry open of the short decided at the close of day 1
    qv = close * 0 + 5e7
    ev = pd.Series({"X": idx[0], "Y": idx[3]}, dtype=object)  # X decided at day 1, Y at day 4
    return idx, opn, close, qv, ev


def test_close_stop_covers_at_the_next_open_frees_the_slot_and_never_reenters():
    idx, opn, close, qv, ev = _stop_case()
    held, stops = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30, max_shorts=1, stop_k=0.5)
    # decided at the close of day 1, entered at the open of day 2 (10.0); the close of day 3 (14.9) is below 15
    assert held["X"].tolist()[:4] == [False, True, True, True]
    # the close of day 4 (15.0) reaches 1.5 x 10: the target is 0 at that close, so the engine covers on day 5
    assert not held["X"].iloc[4:].any()
    assert stops == [{"contract": "X", "decided": idx[1], "stopped": idx[4], "entry_open": 10.0,
                      "stop_close": 15.0, "days_in_position": 3}]
    # the freed slot takes Y, decided at the same close
    assert held["Y"].iloc[4:].all() and not held["Y"].iloc[:4].any()
    # without the stop the slot stays taken, and Y is skipped
    held0, _ = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30, max_shorts=1, stop_k=None)
    assert held0["X"].iloc[1:].all() and not held0["Y"].any()
    # a wider stop is not reached
    held1, stops1 = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30, max_shorts=1, stop_k=1.0)
    pd.testing.assert_frame_equal(held1, held0)
    assert stops1 == []


def test_stop_statistics_use_the_next_open_after_the_stop():
    idx, opn, close, qv, ev = _stop_case()
    _, stops = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30, max_shorts=1, stop_k=0.5)
    s = listing_risk.stop_stats(stops, opn)
    assert s["stops"] == 1 and s["exit_over_entry_median"] == pytest.approx(1.5)  # day 5 opens at 15.0
    assert listing_risk.stop_stats([], opn) == {"stops": 0}


def test_stopped_shorts_use_no_future_data():
    idx, opn, close, qv, ev = _world(seed=8)
    rng = np.random.default_rng(9)
    basket = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])
    cl = close.assign(A=1.0, B=1.0, C=1.0)
    op = opn.assign(A=1.0, B=1.0, C=1.0)
    q = qv.assign(A=1e8, B=1e8, C=1e8)
    base, held, stops = listing_risk.r9a(op, cl, q, ev, basket, 0.5)
    assert len(stops) > 0 and held.any().sum() > 3
    for cut in idx[10::30]:
        later = idx > cut
        c2, o2, q2 = cl.copy(), op.copy(), q.copy()
        shock = np.exp(rng.normal(0, 0.4, c2[later].shape))
        c2[later] = c2[later] * shock
        o2[later] = o2[later] * np.exp(rng.normal(0, 0.4, o2[later].shape))
        q2[later] = rng.uniform(0, 1e8, q2[later].shape)
        w2, _, _ = listing_risk.r9a(o2, c2, q2, ev, basket, 0.5)
        pd.testing.assert_frame_equal(base[idx <= cut], w2[idx <= cut])


def test_r9a_sizes_each_short_at_five_percent_and_stays_dollar_neutral():
    idx, opn, close, qv, ev = _world(seed=11)
    basket = pd.DataFrame(True, index=idx, columns=["A", "B"])
    cl, op, q = close.assign(A=1.0, B=1.0), opn.assign(A=1.0, B=1.0), qv.assign(A=1e8, B=1e8)
    W, held, _ = listing_risk.r9a(op, cl, q, ev, basket, None)
    v = W[held.columns].to_numpy()
    assert set(np.round(v[v != 0], 12)) == {-0.05}
    assert (held.sum(axis=1) <= listing_risk.MAX_SHORTS).all()
    assert W.sum(axis=1).abs().max() < 1e-12


def test_pooled_events_refuse_shared_contracts():
    a = pd.Series({"X": ts("2024-01-02"), "Y": ts("2024-01-01")}, dtype=object)
    b = pd.Series({"Z": ts("2024-01-03")}, dtype=object)
    assert list(listing_risk.pooled_events(a, b).index) == ["Y", "X", "Z"]
    with pytest.raises(ValueError, match="overlap"):
        listing_risk.pooled_events(a, b, pd.Series({"X": ts("2024-02-01")}, dtype=object))


# ------------------------------------------------------------------ runner end to end
def _panel(close: pd.DataFrame, funding: pd.DataFrame) -> Panel:
    o = close.shift(1).fillna(close).where(close.notna())
    return Panel(o, np.maximum(o, close) * 1.001, np.minimum(o, close) * 0.999, close, close * 0 + 5e7, funding,
                 pd.DataFrame(True, index=close.index, columns=close.columns))


@pytest.fixture
def synthetic(monkeypatch):
    from cqs.research import metrics_window
    from cqs.research import run_round9 as R9
    monkeypatch.setattr(metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    rng = np.random.default_rng(1)
    idx = days(1000, "2020-01-01")
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=1), freq="8h", tz=UTC)
    alts = [f"A{i}USDT" for i in range(6)]
    base = {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))) for c in ["BTCUSDT"] + alts}
    lst = {}
    for k, d0 in enumerate(pd.date_range("2020-03-05", "2022-08-01", freq="12D", tz=UTC)):
        c = pd.Series(np.nan, index=idx)
        live = idx >= d0
        c[live] = 5 * np.exp(np.cumsum(rng.normal(-0.002, 0.08, int(live.sum()))))
        lst[f"N{k}USDT"] = (d0, c)
    close = pd.concat([pd.DataFrame(base, index=idx), pd.DataFrame({s: v[1] for s, v in lst.items()})], axis=1)
    fund = pd.DataFrame({s: rng.normal(0, 3e-4, len(ft)) for s in close.columns}, index=ft)
    p = _panel(close, fund)
    basket = pd.DataFrame(False, index=idx, columns=close.columns)
    basket[alts] = True
    ev = pd.Series({s: v[0] for s, v in lst.items()}, dtype=object)
    d = dict(p=p, bars=(p.open, p.high, p.low, p.close), events=ev, basket=basket, rf=pd.Series(0.02, index=idx),
             costs=CostModel(default_slippage_bps=10.0), fingerprint="synthetic", missing_funding=[],
             checked=10, inconsistent=[], events_r7c=40, events_r8b=len(ev) - 40)
    return R9, {"R9A": d}


def test_round9_runs_end_to_end_and_reports_the_stops(synthetic):
    R9, data = synthetic
    out, oos = R9.run_family("R9A", data["R9A"], log=False)
    names = {R9._name(R9.GRIDS["R9A"][0], p) for p in R9.GRIDS["R9A"][1]}
    assert set(out["variants"]) == names and set(out["decisions"]) == names
    for name, pv in out["variants"].items():
        assert "ruined" not in pv and pv["n_trades"] > 0, name
    assert out["decisions"]["r9a_pooled_listing_shorts_stop(stop=None)"]["stops"] == 0
    assert out["decisions"]["r9a_pooled_listing_shorts_stop(stop=0.5)"]["stops"] > 0
    assert out["walk_forward"]["oos"]["start"] == "2022-01-01"
    assert out["events_r7c"] == 40 and len(oos) > 100


def test_round9_main_logs_three_trials_writes_the_summary_and_never_reruns(synthetic, monkeypatch, tmp_path):
    import json

    from cqs import config
    from cqs.research import registry
    R9, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R9, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R9, "preregistered", lambda: True)
    monkeypatch.setattr(R9, "LOADERS", {f: (lambda f=f: data[f], R9.LOADERS[f][1]) for f in R9.LOADERS})
    assert R9.main([]) == 0
    for f in ("R9A.json", "R9A_oos.csv", "R9_summary.json"):
        assert (tmp_path / "results" / f).exists(), f
    ex = [r for r in registry.read() if r.get("type") == "experiment"]
    assert sorted(r["family"] for r in ex) == ["R9A"] * 3
    summ = json.loads((tmp_path / "results" / "R9_summary.json").read_text())
    assert summ["n_trials_program"] == 3 and set(summ["families"]) == {"R9A"}
    assert R9.main([]) == 0
    assert len([r for r in registry.read() if r.get("type") == "experiment"]) == 3


def test_round9_refuses_contracts_without_funding_data(synthetic, monkeypatch, tmp_path):
    from cqs import config
    R9, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R9, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R9, "preregistered", lambda: True)
    bad = data["R9A"] | {"missing_funding": ["N1USDT"]}
    monkeypatch.setattr(R9, "LOADERS", {"R9A": (lambda: bad, R9.LOADERS["R9A"][1])})
    with pytest.raises(SystemExit, match="no funding data"):
        R9.main(["R9A"])
