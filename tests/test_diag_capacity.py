"""R9A capacity diagnostic on cases with known answers: the impact formula, trade classification on an engine run,
and the yearly drag above the slippage already charged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest import engine
from cqs.backtest.costs import CostModel
from cqs.research import diag_capacity_r9a as D


def test_square_root_impact_and_parkinson():
    assert D.impact(0.05, 1e6, 5e7, 0.1) == pytest.approx(0.7 * 0.1 * np.sqrt(1e-3))      # 22.1 bp
    hi = pd.DataFrame({"X": [np.exp(0.1)]})
    lo = pd.DataFrame({"X": [1.0]})
    assert D.parkinson(hi, lo).iloc[0, 0] == pytest.approx(0.1 / np.sqrt(4 * np.log(2)))


def test_trades_are_classified_from_the_target_and_the_position_before():
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    close = pd.DataFrame({"N": [10, 10, 10, 10, 7, 7, 7, 7, 7, 7], "M": 5.0, "B": 3.0}, index=idx, dtype=float)
    opn = close.shift(1).fillna(close)
    high, low = np.maximum(opn, close) * 1.01, np.minimum(opn, close) * 0.99
    W = pd.DataFrame(0.0, index=idx, columns=close.columns)
    W.iloc[2:6, 0] = -0.05                         # N short, decided at the closes of rows 2..5
    W.iloc[2:6, 1] = 0.05                          # M long (a basket member that is also an event contract)
    W.iloc[2:6, 2] = 0.05                          # B long, not an event contract
    res = engine.run(opn, high, low, close, W, CostModel(), rebalance_band_rel=0.2)
    qv = close * 0 + 4e7
    sig = D.parkinson(high, low)
    T = D.short_trades(res.fills, W, res.positions, res.equity, qv, sig, ["N", "M"])
    assert list(T["asset"]) == ["N"] * 3
    assert list(T["kind"]) == ["entry", "rebalance", "exit"]       # N fell 30% on row 4: the band refills it
    assert list(T["time"]) == [idx[3], idx[5], idx[7]]
    assert T["w"].iloc[0] == pytest.approx(0.05)
    assert T["qv"].eq(4e7).all() and T["sigma"].iloc[0] == pytest.approx(sig.at[idx[3], "N"])


def test_drag_counts_only_impact_above_the_charged_slippage():
    t = pd.to_datetime(["2023-01-01", "2023-06-01"], utc=True)
    T = pd.DataFrame({"time": t, "asset": "N", "kind": ["entry", "exit"], "w": 0.05, "qv": 5e7, "sigma": 0.1})
    out = D.summarize(T, equities=(1e6,), end=pd.Timestamp("2024-01-01", tz="UTC"),
                      oos_start=pd.Timestamp("2023-01-01", tz="UTC"))
    v = out["full"]["by_equity"]["1000000"]
    imp = 0.7 * 0.1 * np.sqrt(1e-3)
    assert v["weighted_mean_impact_bp"] == pytest.approx(imp * 1e4)
    assert v["drag_above_charged_per_year"] == pytest.approx(2 * 0.05 * (imp - 1e-3) / (365 / 365.25))
    assert out["full"]["short_leg_turnover_per_year"] == pytest.approx(0.1 / (365 / 365.25))
    assert out["by_kind"]["entry"]["n"] == 1 and out["dropped_no_volume_or_range"] == 0
