"""The R9A final-test runner on synthetic data only: its locks, its pass rule and one run end to end.
No test here reads the real holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest.costs import CostModel
from cqs.research.panel import Panel

UTC = "UTC"


def _series(n, mu, sd, seed, start="2025-06-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.date_range(start, periods=n, freq="1D", tz=UTC))


def test_verdict_applies_the_three_preregistered_conditions():
    from cqs.research import final_r9a as F
    good = _series(500, 0.002, 0.01, 0)
    v = F.verdict(good, good - 0.0005)
    assert v["passed"] and all(v["checks"].values())
    crash = good.copy()
    crash.iloc[200:215] = -0.03                      # about -37% in three weeks
    v = F.verdict(crash, crash)
    assert not v["checks"]["max_dd_within_25pct"] and not v["passed"]
    weak = _series(500, 0.0002, 0.01, 1)             # Sharpe well below 1
    assert not F.verdict(weak, weak)["checks"]["sharpe_ge_1"]
    v = F.verdict(good, good - 0.01)                  # loses at costs x2
    assert not v["checks"]["positive_at_costs_x2"] and not v["passed"]


def test_halves_split_the_holdout_by_half_year():
    from cqs.research import final_r9a as F
    h = F.halves(pd.Series(0.001, index=pd.date_range("2025-06-01", "2026-02-10", freq="1D", tz=UTC)))
    assert list(h) == ["2025H1", "2025H2", "2026H1"] and h["2025H1"]["days"] == 30


def _panel(close: pd.DataFrame, funding: pd.DataFrame) -> Panel:
    o = close.shift(1).fillna(close).where(close.notna())
    return Panel(o, np.maximum(o, close) * 1.001, np.minimum(o, close) * 0.999, close, close * 0 + 5e7, funding,
                 pd.DataFrame(True, index=close.index, columns=close.columns))


@pytest.fixture
def world(monkeypatch, tmp_path):
    from cqs import config
    from cqs.research import final_r9a as F
    from cqs.research import metrics_window, registry
    monkeypatch.setattr(metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(F, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(F, "preregistered", lambda: True)
    monkeypatch.setattr(registry, "_code_version", lambda: {"commit": "test", "dirty": False})
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", "2025-09-30", freq="1D", tz=UTC)
    ft = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=1), freq="8h", tz=UTC)
    alts = [f"A{i}USDT" for i in range(5)]
    base = {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))) for c in ["BTCUSDT"] + alts}
    lst = {}
    for k, d0 in enumerate(pd.date_range("2024-01-10", "2025-08-20", freq="9D", tz=UTC)):
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
             checked=0, inconsistent=[], events_r7c=set(ev.index[::2]), events_r8b=set(ev.index[1::2]))
    return F, d


def _events():
    from cqs.research import registry
    return [r.get("event") for r in registry.read() if r.get("type") == "event"]


def test_final_test_needs_the_holdout_key_and_leaves_no_trace_without_it(world, monkeypatch):
    from cqs.data import guard
    F, d = world
    monkeypatch.delenv("CQS_ALLOW_HOLDOUT", raising=False)
    with pytest.raises(guard.HoldoutAccessError):
        F.main(["--approval", "test approval, 2026-11-26"], loader=lambda: d)
    assert _events() == []


def test_final_test_refuses_without_preregistration_approval_or_clean_code(world, monkeypatch):
    from cqs.research import registry
    F, d = world
    monkeypatch.setenv("CQS_ALLOW_HOLDOUT", "1")
    with pytest.raises(SystemExit, match="quote the owner"):
        F.main(["--approval", "ok"], loader=lambda: d)
    monkeypatch.setattr(registry, "_code_version", lambda: {"commit": "test", "dirty": True})
    with pytest.raises(SystemExit, match="commit the code"):
        F.main(["--approval", "test approval, 2026-11-26"], loader=lambda: d)
    monkeypatch.setattr(F, "preregistered", lambda: False)
    with pytest.raises(SystemExit, match="not pre-registered"):
        F.main(["--approval", "test approval, 2026-11-26"], loader=lambda: d)
    assert _events() == []


def test_final_test_runs_once_scores_only_the_holdout_and_logs_everything(world, monkeypatch):
    import json

    from cqs import config
    F, d = world
    monkeypatch.setenv("CQS_ALLOW_HOLDOUT", "1")
    assert F.main(["--approval", "test approval, 2026-11-26"], loader=lambda: d) == 0
    assert _events() == ["holdout_opened", F.EVENT, "holdout_closed"]
    out = json.loads((F.RESULTS_DIR / "FINAL_R9A.json").read_text())
    assert out["holdout"]["start"] == str(config.HOLDOUT_START.date())
    assert set(out["checks"]) == {"sharpe_ge_1", "max_dd_within_25pct", "positive_at_costs_x2"}
    assert out["passed"] == all(out["checks"].values())
    assert out["diagnostics"]["events_with_d0_in_holdout"] > 0
    h = pd.read_csv(F.RESULTS_DIR / "FINAL_R9A_holdout.csv", index_col=0, parse_dates=True)
    assert h.index[0] >= config.HOLDOUT_START and len(h) == out["holdout"]["days"]
    with pytest.raises(SystemExit, match="never repeated"):
        F.main(["--approval", "test approval, 2026-11-26"], loader=lambda: d)
