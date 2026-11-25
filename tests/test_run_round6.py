"""Round 6 runner: both families and the second-venue confirmation run end to end on synthetic data, without
touching the registry; the T-bill charge and the execution check behave as specified."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.data.quarterly import Contract
from cqs.research import run_round6 as R6

UTC = "UTC"
DELIVERIES = ["2021-03-26", "2021-06-25", "2021-09-24", "2021-12-31", "2022-03-25", "2022-06-24", "2022-09-30"]


def _venue(names: dict, hourly_funding: bool, seed: int = 0) -> R6.Venue:
    """names: perpetual column -> quarterly name pattern ({d} = yymmdd)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", "2022-07-01", freq="1h", tz=UTC, inclusive="left")
    raw, legs, funding = {}, {}, {}
    for k, (perp, pattern) in enumerate(names.items()):
        p = 30000.0 / (1 + 14 * k) * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
        raw[perp] = p
        b = 0.06 + 0.04 * np.sin(np.arange(len(idx)) / 900 + k) + rng.normal(0, 0.01, len(idx))
        legs[perp] = []
        for d in DELIVERIES:
            dl = pd.Timestamp(d, tz=UTC) + pd.Timedelta(hours=8)
            c = Contract(pattern.format(d=dl.strftime("%y%m%d")), max(dl - pd.Timedelta(days=120), idx[0]), dl)
            D = ((dl - idx) / pd.Timedelta(days=1)).to_numpy()
            raw[c.symbol] = np.where((idx >= c.listed) & (idx < dl), p * (1 + b * D / 365), np.nan)
            legs[perp].append(c)
        step = "1h" if hourly_funding else "8h"
        ft = pd.date_range(pd.Timestamp("2020-11-01", tz=UTC), idx[-1], freq=step)
        per = 1 / 24 if hourly_funding else 1 / 3
        funding[perp] = pd.Series((0.05 + rng.normal(0, 0.05, len(ft))) / 365 * per, index=ft)
    C = pd.DataFrame(raw, index=idx)
    O = C.shift(1).fillna(C)  # open = previous close (no gaps inside a contract's life)
    O = O.where(C.notna())
    H = np.maximum(O, C) * 1.001
    L = np.minimum(O, C) * 0.999
    return R6.Venue("synthetic", (O, H, L, C), pd.DataFrame(funding), legs, "synthetic")


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(R6.metrics_window, "benchmarks_in_window", lambda *a, **k: {})
    v = _venue({"BTCUSDT": "BTCUSDT_{d}", "ETHUSDT": "ETHUSDT_{d}"}, hourly_funding=False)
    vd = _venue({"BTC-PERPETUAL": "BTC-{d}", "ETH-PERPETUAL": "ETH-{d}"}, hourly_funding=True, seed=1)
    rf = pd.Series(0.02, index=pd.date_range("2020-11-01", "2022-07-01", freq="1D", tz=UTC))
    dclose = v.bars[3]["BTCUSDT"].resample("1D").last()
    return v, vd, rf, dclose


def test_every_round6_family_and_the_confirmation_run_end_to_end(setup):
    v, vd, rf, dclose = setup
    for fam, (fname, grid) in R6.GRIDS.items():
        out, oos = R6.run_family(fam, v, rf, dclose, log=False)
        assert set(out["variants"]) == {R6._name(fname, p) for p in grid}, fam
        for name, pv in out["variants"].items():
            assert "ruined" not in pv, (name, pv)
            assert pv["n_trades"] > 0, name
            d = pv["decisions"]["BTCUSDT"]
            assert d["changes"] > 0 and d["share_in_position"] > 0, name
        assert out["walk_forward"]["oos"]["start"] == "2022-01-01", fam
        assert len(oos) > 100
        assert ("pbo_note" in out) == (len(grid) == 1)
        conf = R6.confirm(fam, out["walk_forward"]["choices"], vd, rf, log=False)
        assert set(conf) >= {"BTC+ETH", "BTC", "ETH", "passed"}
        assert conf["BTC+ETH"]["oos"]["days"] == len(oos)


def test_r6a_is_always_short_basis_and_rolls_every_quarter(setup):
    v, _, rf, _ = setup
    W, feats = R6.build_weights(v, "R6A")
    f = feats["BTCUSDT"]
    assert set(np.unique(f["state"])) == {1}  # each contract is listed 120 days before delivery: always a front
    stats = R6._decision_stats(feats)["BTCUSDT"]
    assert stats["rolls_in_position"] == 6  # out of 2021-03, 2021-06, 2021-09, 2021-12, 2022-03 and 2022-06
    assert stats["days_without_front"] == 0
    # the perpetual is long and the quarterlies short whenever a weight is set
    assert (W["BTCUSDT"].dropna() == 0.5).all()
    q = W[[c.symbol for c in v.legs["BTCUSDT"]]]
    assert set(np.unique(q.stack().dropna())) <= {0.0, -0.5}


def test_tbill_is_charged_on_the_margin_only_while_a_position_is_open():
    idx = pd.date_range("2022-01-01", periods=72, freq="1h", tz=UTC)
    pos = pd.DataFrame(0.0, index=idx, columns=["A", "B", "C", "D"])
    pos.iloc[24:48] = [0.5, -0.5, 0.5, -0.5]  # day 2: gross 2x -> margin = the whole equity
    pos.iloc[48:] = [0.5, -0.5, 0.0, 0.0]  # day 3: one coin -> half the equity
    res = type("R", (), {"positions": pos, "equity": pd.Series(1.0, index=idx)})()
    rf = pd.Series(0.0365, index=pd.date_range("2022-01-01", periods=3, freq="1D", tz=UTC))
    charge = R6.tbill_on_margin(res, rf)
    assert charge.tolist() == pytest.approx([0.0, 0.0001, 0.00005])


def test_execution_check_refuses_a_one_legged_trade(setup):
    v, _, _, _ = setup
    W, _ = R6.build_weights(v, "R6A")
    i = int(np.flatnonzero(W.notna().any(axis=1).to_numpy())[0])
    col = W.columns[(W.iloc[i].notna()) & (W.iloc[i] != 0)][-1]
    O = v.bars[0].copy()
    O.iloc[i + 1, O.columns.get_loc(col)] = np.nan
    broken = R6.Venue(v.name, (O, *v.bars[1:]), v.funding, v.legs, v.fingerprint)
    with pytest.raises(RuntimeError, match="no execution bar"):
        R6.check_execution(broken, W)
