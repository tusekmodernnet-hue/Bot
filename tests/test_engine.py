import numpy as np
import pandas as pd
import pytest

from cqs.backtest import engine
from cqs.backtest.costs import CostModel
from tests.conftest import make_bars

ZERO = CostModel(taker_fee=0.0, default_slippage_bps=0.0, forced_exit_slippage_bps=0.0)
C10 = CostModel(taker_fee=0.001, default_slippage_bps=0.0, forced_exit_slippage_bps=0.0)


def weights_like(close, values):
    return pd.DataFrame({"A": np.asarray(values, float)}, index=close.index)


def test_buy_and_hold_matches_price_return_minus_entry_cost():
    o, h, l, c = make_bars([100, 110, 121, 133.1, 120])
    w = weights_like(c, [1, np.nan, np.nan, np.nan, np.nan])  # NaN = hold, no re-balancing
    res = engine.run(o, h, l, c, w, C10)
    # entry at open of bar 1 (=100): pay 0.1% of 1.0, hold 1.0 notional to the last close (120)
    assert res.equity.iloc[0] == 1.0
    assert res.equity.iloc[-1] == pytest.approx((1 - 0.001) + 1.0 * (120 / 100 - 1))
    assert len(res.trades) == 1 and res.trades.exit_reason.iloc[0] == "end_of_data"


def test_execution_is_next_bar_open_not_same_bar():
    # weight set at close of bar 2 must not earn bar 2's move
    o, h, l, c = make_bars([100, 100, 150, 150, 150])
    w = weights_like(c, [0, 0, 1, 1, 1])
    res = engine.run(o, h, l, c, w, ZERO)
    assert res.equity.iloc[-1] == pytest.approx(1.0)
    assert res.positions["A"].iloc[2] == 0 and res.positions["A"].iloc[3] != 0


def test_gap_at_open_is_borne_by_old_position():
    # close 100 -> next open 90 (gap), close 95. Long entered at open of bar 1 at 100 (no gap there).
    o, h, l, c = make_bars([100, 100, 95], opens=[100, 100, 90])
    w = weights_like(c, [1, 1, 1])
    res = engine.run(o, h, l, c, w, ZERO)
    assert res.equity.iloc[-1] == pytest.approx(0.95)


def test_funding_long_pays_short_receives():
    o, h, l, c = make_bars([100] * 10)
    idx = c.index
    funding = pd.DataFrame({"A": [0.001, 0.001]}, index=[idx[3], idx[6] - pd.Timedelta(minutes=30)])
    for sign in (1, -1):
        res = engine.run(o, h, l, c, weights_like(c, [sign] + [np.nan] * 9), ZERO, funding=funding)
        assert res.equity.iloc[-1] == pytest.approx(1 - sign * 0.002)
        assert res.funding_paid == pytest.approx(sign * 0.002)


def test_funding_before_entry_is_not_charged():
    o, h, l, c = make_bars([100] * 6)
    funding = pd.DataFrame({"A": [0.01]}, index=[c.index[1]])  # settles at open of bar 1
    w = weights_like(c, [0, 1, 1, 1, 1, 1])  # first position opens at open of bar 2
    res = engine.run(o, h, l, c, w, ZERO, funding=funding)
    assert res.funding_paid == 0.0


def test_stop_fills_at_stop_or_gap_open_and_stays_flat_until_target_changes():
    closes = [100, 100, 100, 100, 100, 100]
    opens = [100, 100, 100, 100, 100, 100]
    lows = [100, 100, 94, 100, 100, 100]  # bar 2 trades down through the stop at 95
    o, h, l, c = make_bars(closes, opens=opens, lows=lows)
    w = weights_like(c, [1, 1, 1, 1, 2, 2])  # target changes at close of bar 4 -> re-entry at bar 5
    stops = weights_like(c, [95] * 6)
    res = engine.run(o, h, l, c, w, ZERO, stops=stops)
    assert res.positions["A"].iloc[2] == 0 and res.positions["A"].iloc[3] == 0
    assert res.positions["A"].iloc[5] != 0
    assert res.equity.iloc[2] == pytest.approx(1 - 0.05)
    assert list(res.trades.exit_reason)[:1] == ["stop"]
    # gap through the stop: fill at the open, not at the stop price
    o2, h2, l2, c2 = make_bars([100, 100, 90], opens=[100, 100, 90], lows=[100, 100, 88])
    res2 = engine.run(o2, h2, l2, c2, weights_like(c2, [1, 1, 1]), ZERO, stops=weights_like(c2, [95, 95, 95]))
    assert res2.equity.iloc[-1] == pytest.approx(0.90)


def test_reenter_after_stop_takes_a_new_entry_of_the_same_size():
    lows = [100, 100, 94, 100, 100, 100]
    o, h, l, c = make_bars([100] * 6, opens=[100] * 6, lows=lows)
    # entry at close 1, stopped inside bar 2, a new entry of the same size decided at close 2
    w = weights_like(c, [np.nan, 1, 1, np.nan, np.nan, np.nan])
    stops = weights_like(c, [np.nan, 95, 95, 95, 95, 95])
    held = engine.run(o, h, l, c, w, ZERO, stops=stops).positions["A"]
    assert held.iloc[3] == 0  # default: the same target after a stop keeps the asset flat
    held = engine.run(o, h, l, c, w, ZERO, stops=stops, reenter_after_stop=True).positions["A"]
    assert held.iloc[2] == 0 and held.iloc[3] != 0
    # a hold (NaN) after the stop still keeps it flat
    w2 = weights_like(c, [np.nan, 1, np.nan, np.nan, np.nan, np.nan])
    held = engine.run(o, h, l, c, w2, ZERO, stops=stops, reenter_after_stop=True).positions["A"]
    assert (held.iloc[2:] == 0).all()


def test_short_stop_and_costs_on_exit():
    o, h, l, c = make_bars([100, 100, 100], highs=[100, 100, 106])
    res = engine.run(o, h, l, c, weights_like(c, [-1, np.nan, np.nan]), C10, stops=weights_like(c, [105] * 3))
    entry_cost = 0.001
    exit_cost = 1.05 * 0.001
    assert res.equity.iloc[-1] == pytest.approx(1 - entry_cost - 0.05 - exit_cost)


def test_delisting_forces_exit_at_last_close_and_weight_on_missing_bar_raises():
    o, h, l, c = make_bars([100, 100, 110, np.nan, np.nan])
    o.iloc[3:] = np.nan
    cm = CostModel(taker_fee=0.0, default_slippage_bps=0.0, forced_exit_slippage_bps=200.0)
    with pytest.raises(ValueError):  # a weight decided on a bar that has no data is a strategy bug
        engine.run(o, h, l, c, weights_like(c, [1, 1, 1, 1, 0]), cm)
    w = weights_like(c, [1, np.nan, np.nan, 0, 0])
    res = engine.run(o, h, l, c, w, cm)
    pos_at_exit = 1.1  # 1.0 notional bought at 100, marked to 110
    assert res.equity.iloc[-1] == pytest.approx(1.0 + 0.1 - pos_at_exit * 0.02)
    assert res.trades.exit_reason.iloc[0] == "forced_exit"


def test_flip_creates_two_trades_with_split_costs():
    o, h, l, c = make_bars([100] * 5)
    res = engine.run(o, h, l, c, weights_like(c, [1, 1, -1, -1, -1]), C10)
    assert list(res.trades.side) == ["long", "short"]
    assert res.trades.pnl.iloc[0] == pytest.approx(-0.002)  # entry 1.0*0.1% + exit 1.0*0.1%
    assert res.costs_paid == pytest.approx(res.traded_notional * 0.001)


def test_costs_scale_with_turnover():
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    o, h, l, c = make_bars(closes)
    w = weights_like(c, np.sign(rng.normal(size=300)))
    a = engine.run(o, h, l, c, w, ZERO)
    b = engine.run(o, h, l, c, w, C10)
    assert b.equity.iloc[-1] < a.equity.iloc[-1]
    assert b.costs_paid == pytest.approx(b.traded_notional * 0.001)


def test_weights_drift_between_rebalances():
    # weight 0.5 decided once; price doubles; no rebalance trade unless weight is re-asserted each bar
    o, h, l, c = make_bars([100, 100, 200, 200])
    res = engine.run(o, h, l, c, weights_like(c, [0.5, 0.5, 0.5, 0.5]), ZERO)
    # after doubling, equity 1.5 and target 0.75 -> engine rebalances from 1.0 to 0.75 at bar 3 open
    assert res.positions["A"].iloc[2] == pytest.approx(1.0)
    assert res.positions["A"].iloc[3] == pytest.approx(0.75)


def test_rebalance_band_skips_small_trades():
    o, h, l, c = make_bars([100, 101, 102, 101, 100])
    exact = engine.run(o, h, l, c, weights_like(c, [1] * 5), C10)
    banded = engine.run(o, h, l, c, weights_like(c, [1] * 5), C10, rebalance_band=0.05)
    assert len(banded.fills) == 1 and len(exact.fills) > 1
    assert banded.costs_paid < exact.costs_paid
