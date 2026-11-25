"""Round 5 runner: every family runs end to end on synthetic data, without touching the registry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.research import run_round5 as R5

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "taker_buy_volume": "sum"}


def _asset(seed, days=150, px=30000.0):
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2021-01-01", periods=n, freq="1min", tz="UTC")
    r = 0.0008 * rng.standard_t(3, n) / np.sqrt(3)
    c = px * np.exp(np.cumsum(r))
    o = np.r_[px, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0004, n)))
    v = rng.uniform(5, 15, n)
    share = np.clip(0.5 + 150 * r + rng.normal(0, 0.1, n), 0.02, 0.98)
    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "taker_buy_volume": v * share},
                      index=idx)
    m15 = m1.resample("15min").agg(AGG)
    h1 = m1.resample("1h").agg(AGG)
    t = pd.date_range(idx[0], idx[-1], freq="5min")
    oi = pd.DataFrame({"create_time": t, "available_time": t + pd.Timedelta(minutes=5),
                       "sum_open_interest": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.006, len(t))))})
    return m1[["open", "high", "low", "close"]], m15, h1, oi


def _data():
    m1, m15, h1, oi = {}, {}, {}, {}
    for k, s in enumerate(R5.MAJORS):
        m1[s], m15[s], h1[s], oi[s] = _asset(20 + k, px=30000.0 / (k * 14 + 1))
    idx = m15[R5.MAJORS[0]].index
    bars = tuple(pd.DataFrame({s: m15[s][f] for s in R5.MAJORS}).reindex(idx) for f in ("open", "high", "low", "close"))
    ft = pd.date_range(idx[0], idx[-1], freq="8h")
    funding = pd.DataFrame({s: np.full(len(ft), 1e-4) for s in R5.MAJORS}, index=ft)
    dclose = m15["BTCUSDT"]["close"].resample("1D").last()
    return R5.Data(m15, h1, oi, dclose, idx, bars, funding, m1)


def test_every_round5_family_runs_end_to_end(monkeypatch):
    monkeypatch.setattr(R5, "fingerprint", lambda: "synthetic")
    monkeypatch.setattr(R5.metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    d = _data()
    for fam, (fname, _, grid) in R5.GRIDS.items():
        out, oos = R5.run_family(fam, d, log=False)
        assert set(out["variants"]) == {R5._name(fname, p) for p in grid}, fam
        for name, v in out["variants"].items():
            assert "ruined" not in v, name
            assert v["n_trades"] > 0, f"{name} never trades on synthetic data"
            assert "engine_trade_mismatch" not in v, name  # every decision reached the engine
        assert np.isfinite(out["pbo"]), fam
