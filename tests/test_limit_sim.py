"""Round 4: limit-order fill rules on hand-built bars, and no look-ahead in the order plans."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs.backtest.limit_sim import Fees, Plans, simulate
from cqs.strategies import intraday as I
from cqs.strategies import scalping as S

ZERO = Fees(0.0, 0.0, 0.0)


def _bars(rows):
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    idx = pd.date_range("2024-01-01", periods=len(o), freq="1min", tz="UTC")
    return idx, o, h, l, c


def _plan(start, end, bp=np.nan, btp=np.nan, bsl=np.nan, sp=np.nan, stp=np.nan, ssl=np.nan, da=-1, dr=-1):
    a = lambda x: np.array([x])  # noqa: E731
    return Plans(a(start), a(end), a(bp), a(btp), a(bsl), a(sp), a(stp), a(ssl), a(da), a(dr))


def test_limit_fills_only_when_price_trades_through():
    rows = [(100, 100.5, 99.0, 100)] * 2 + [(100, 101, 99.5, 100)] * 3
    idx, o, h, l, c = _bars(rows)
    eq, t = simulate(idx, o, h, l, c, _plan(0, 4, bp=99.0, btp=101.0, bsl=98.0), ZERO)
    assert t.empty and eq.iloc[-1] == 1.0  # low touched 99.0 but never traded below it
    eq, t = simulate(idx, o, h, l, c, _plan(0, 4, bp=99.2, btp=100.8, bsl=98.2), ZERO)
    assert len(t) == 1 and t.reason.iloc[0] == "target"
    u = 0.005 / (1.0 / 99.2) / 99.2  # 0.5% risk over a 1.0 stop distance, in units
    assert eq.iloc[-1] == pytest.approx(1 + u * (100.8 - 99.2))


def test_stop_wins_when_target_and_stop_share_a_bar_and_fees_apply():
    rows = [(100, 100, 99.0, 99.5), (99.5, 101.5, 97.0, 100)]
    idx, o, h, l, c = _bars(rows)
    fees = Fees(maker=0.0002, taker=0.0005, slippage=0.0002)
    eq, t = simulate(idx, o, h, l, c, _plan(0, 1, bp=99.5, btp=101.0, bsl=98.5), fees)
    assert list(t.reason) == ["stop"]
    u = 0.005 / (1.0 / 99.5) / 99.5
    out = 98.5 * (1 - 0.0002)
    expected = 1 - u * 99.5 * 0.0002 + u * (out - 99.5) - u * out * 0.0005
    assert eq.iloc[-1] == pytest.approx(expected)


def test_stop_in_the_entry_bar_and_gap_through_stop():
    idx, o, h, l, c = _bars([(100, 100, 97.0, 99.0), (99, 99, 98, 98.5)])
    eq, t = simulate(idx, o, h, l, c, _plan(0, 1, bp=99.5, btp=101.0, bsl=98.5), ZERO)
    assert list(t.reason) == ["stop"] and t.exit.iloc[0] == 98.5
    idx, o, h, l, c = _bars([(100, 100, 99.0, 99.6), (97.0, 97.5, 96.5, 97.0)])
    eq, t = simulate(idx, o, h, l, c, _plan(0, 0, bp=99.5, btp=101.0, bsl=98.5), ZERO)
    assert list(t.reason) == ["stop"] and t.exit.iloc[0] == 97.0  # gap: filled at the open


def test_time_exit_at_the_open_after_the_deadline_and_one_plan_at_a_time():
    rows = [(100, 100, 99.0, 99.6)] + [(99.6 + k * 0.1, 99.8 + k * 0.1, 99.55 + k * 0.1, 99.7 + k * 0.1) for k in range(6)]
    idx, o, h, l, c = _bars(rows)
    eq, t = simulate(idx, o, h, l, c, _plan(0, 6, bp=99.5, btp=200.0, bsl=98.5, dr=2), ZERO)
    assert list(t.reason) == ["time"] and t.exit_time.iloc[0] == idx[3] and t.exit.iloc[0] == o[3]
    # a short side in the same bar as the long: the worse of the two is assumed
    idx, o, h, l, c = _bars([(100, 101.0, 99.0, 100.8), (100.8, 100.9, 100.7, 100.8)])
    eq, t = simulate(idx, o, h, l, c, _plan(0, 1, bp=99.5, btp=105, bsl=98.5, sp=100.5, stp=95, ssl=101.5), ZERO)
    assert list(t.side) == [-1]  # the short marks a loss at the close of 100.8, the long a gain


def _m1(days=45, seed=4):
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC")
    c = 40000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.uniform(1, 9, n)}, index=idx)


def _agg(m1, rule):
    return I.resample(m1, rule)


def _perturb_after(df, t, seed=7):
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = out.index > t
    f = np.exp(np.cumsum(rng.normal(0, 0.002, m.sum()))) * 1.03
    for col in ("open", "high", "low", "close"):
        out.loc[m, col] = out.loc[m, col].to_numpy() * f
    return out


def _plan_frame(p: Plans) -> pd.DataFrame:
    return pd.DataFrame({k: getattr(p, k) for k in p.__dataclass_fields__})


def test_order_plans_use_no_future_data():
    m1 = _m1()
    for pos in (40 * 1440 + 7 * 60 + 2, 41 * 1440 + 13 * 60 + 37):
        t = m1.index[pos]  # orders starting at or before t must not depend on bars after t
        m1b = _perturb_after(m1, t - pd.Timedelta(minutes=1))
        for build in (lambda m: S.s1_plans(_agg(m, "5min"), _agg(m, "1h"), m.index, 2.0, False),
                      lambda m: S.s1_plans(_agg(m, "5min"), _agg(m, "1h"), m.index, 2.0, True),
                      lambda m: S.s2_plans(_agg(m, "5min"), _agg(m, "1h"), m.index, 0.5)):
            a, b = _plan_frame(build(m1)), _plan_frame(build(m1b))
            a, b = a[a.start <= pos].reset_index(drop=True), b[b.start <= pos].reset_index(drop=True)
            assert len(a) > 0
            pd.testing.assert_frame_equal(a, b)
