"""Round 12 building blocks on cases with known answers, a look-ahead mutation test and the runner end to end on
synthetic data (R12A: the R9A shorts decided at the close of the listing day)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest.costs import CostModel
from cqs.research.panel import Panel
from cqs.strategies import listing_risk

UTC = "UTC"


def days(n, start="2022-01-01"):
    return pd.date_range(start, periods=n, freq="1D", tz=UTC)


def _case():
    idx = days(40, "2024-03-01")
    close = pd.DataFrame({"X": np.r_[np.full(3, np.nan), np.full(37, 10.0)],
                          "Y": np.r_[np.full(5, np.nan), np.full(35, 4.0)]}, index=idx)
    opn = close.shift(1).fillna(close).where(close.notna())
    qv = (close * 0 + 5e7)
    qv.loc[idx[5], "Y"] = 1e7                  # Y's listing day trades only 10M USDT, the next day 50M
    ev = pd.Series({"X": idx[3], "Y": idx[5]}, dtype=object)
    return idx, opn, close, qv, ev


def test_listing_day_entry_uses_the_listing_day_volume():
    idx, opn, close, qv, ev = _case()
    held, stops = listing_risk.shorts_with_stop(opn, close, qv, ev, 0, 30, liq_from=0)
    x = held["X"]
    assert not x.iloc[:3].any() and x.iloc[3:33].all() and not x.iloc[33:].any()   # decided at the close of D0
    assert not held["Y"].any() and stops == []                                     # 10M on the listing day
    r9a, _ = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30)               # the R9A rule, unchanged
    assert r9a["X"].iloc[4:34].all() and not r9a["X"].iloc[:4].any() and r9a["Y"].iloc[6:36].all()


def test_the_default_volume_window_is_the_r9a_one():
    idx, opn, close, qv, ev = _case()
    a, _ = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30, stop_k=1.0)
    b, _ = listing_risk.shorts_with_stop(opn, close, qv, ev, 1, 30, stop_k=1.0, liq_from=1)
    pd.testing.assert_frame_equal(a, b)


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
    opn = close.shift(1).where(close.notna()).fillna(close)
    qv = pd.DataFrame(rng.uniform(5e6, 8e7, close.shape), index=idx, columns=close.columns).where(close.notna())
    return idx, opn, close, qv, pd.Series(ev, dtype=object)


def test_r12a_uses_no_future_data():
    idx, opn, close, qv, ev = _world(seed=8)
    rng = np.random.default_rng(10)
    basket = pd.DataFrame(True, index=idx, columns=["A", "B", "C"])
    cl, op, q = close.assign(A=1.0, B=1.0, C=1.0), opn.assign(A=1.0, B=1.0, C=1.0), qv.assign(A=1e8, B=1e8, C=1e8)
    base, held, stops = listing_risk.r9a(op, cl, q, ev, basket, 1.0, delay=0, liq_from=0)
    assert held.any().sum() > 3
    for cut in idx[10::30]:
        later = idx > cut
        c2, o2, q2 = cl.copy(), op.copy(), q.copy()
        c2[later] = c2[later] * np.exp(rng.normal(0, 0.4, c2[later].shape))
        o2[later] = o2[later] * np.exp(rng.normal(0, 0.4, o2[later].shape))
        q2[later] = rng.uniform(0, 1e8, q2[later].shape)
        w2, _, _ = listing_risk.r9a(o2, c2, q2, ev, basket, 1.0, delay=0, liq_from=0)
        pd.testing.assert_frame_equal(base[idx <= cut], w2[idx <= cut])


# ------------------------------------------------------------------ runner end to end
def _panel(close: pd.DataFrame, funding: pd.DataFrame) -> Panel:
    o = close.shift(1).fillna(close).where(close.notna())
    return Panel(o, np.maximum(o, close) * 1.001, np.minimum(o, close) * 0.999, close, close * 0 + 5e7, funding,
                 pd.DataFrame(True, index=close.index, columns=close.columns))


@pytest.fixture
def synthetic(monkeypatch):
    from cqs.research import metrics_window
    from cqs.research import run_round12 as R12
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
    return R12, {"R12A": d}


def test_round12_runs_end_to_end(synthetic):
    R12, data = synthetic
    out, oos = R12.run_family("R12A", data["R12A"], log=False)
    names = {R12._name(R12.GRIDS["R12A"][0], p) for p in R12.GRIDS["R12A"][1]}
    assert set(out["variants"]) == names and set(out["decisions"]) == names
    for name, pv in out["variants"].items():
        assert "ruined" not in pv and pv["n_trades"] > 0, name
    assert out["decisions"]["r12a_listing_day_entry(stop=None)"]["stops"] == 0
    assert out["walk_forward"]["oos"]["start"] == "2022-01-01" and len(oos) > 100


def test_round12_main_logs_two_trials_and_never_reruns(synthetic, monkeypatch, tmp_path):
    import json

    from cqs import config
    from cqs.research import registry
    R12, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R12, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R12, "preregistered", lambda: True)
    monkeypatch.setattr(R12, "LOADERS", {f: (lambda f=f: data[f], R12.LOADERS[f][1]) for f in R12.LOADERS})
    assert R12.main([]) == 0
    for f in ("R12A.json", "R12A_oos.csv", "R12_summary.json"):
        assert (tmp_path / "results" / f).exists(), f
    ex = [r for r in registry.read() if r.get("type") == "experiment"]
    assert sorted(r["family"] for r in ex) == ["R12A"] * 2
    summ = json.loads((tmp_path / "results" / "R12_summary.json").read_text())
    assert summ["n_trials_program"] == 2
    assert R12.main([]) == 0
    assert len([r for r in registry.read() if r.get("type") == "experiment"]) == 2


def test_round12_refuses_contracts_without_funding_data(synthetic, monkeypatch, tmp_path):
    from cqs import config
    R12, data = synthetic
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R12, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R12, "preregistered", lambda: True)
    bad = data["R12A"] | {"missing_funding": ["N1USDT"]}
    monkeypatch.setattr(R12, "LOADERS", {"R12A": (lambda: bad, R12.LOADERS["R12A"][1])})
    with pytest.raises(SystemExit, match="no funding data"):
        R12.main(["R12A"])
