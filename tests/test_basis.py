"""Round 6 building blocks on cases with known answers: delivery dates, the front contract and its roll, the
annualised basis, the trailing funding with its 5-minute delay, the R6B state machine, the weights handed to the
engine, the engine P&L of a hedged position and the absence of look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqs import config
from cqs.backtest import engine
from cqs.backtest.costs import CostModel
from cqs.data import deribit, quarterly
from cqs.data.quarterly import Contract
from cqs.strategies import basis

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


def test_binance_delivery_and_zombie_cut():
    assert quarterly.delivery("BTCUSDT_230929") == ts("2023-09-29 08:00")
    bars = pd.DataFrame({"open_time": pd.date_range("2023-09-29 05:00", periods=6, freq="1h", tz=UTC)})
    cut = quarterly.cut_at_delivery(bars, "BTCUSDT_230929")
    assert cut["open_time"].iloc[-1] == ts("2023-09-29 07:00")  # the 07:00 bar closes at delivery
    with pytest.raises(ValueError):
        quarterly.delivery("BTCUSDT")


def test_deribit_quarterly_names_follow_the_last_friday_rule():
    assert deribit.last_friday(2022, 3).isoformat() == "2022-03-25"
    assert deribit.quarterly_name("BTC", 2022, 3) == "BTC-25MAR22"
    assert deribit.delivery("BTC-25MAR22") == ts("2022-03-25 08:00")
    assert deribit.quarterly_names("ETH", "2021-03-01", "2021-12-31") == [
        "ETH-26MAR21", "ETH-25JUN21", "ETH-24SEP21", "ETH-31DEC21"]


def test_trailing_funding_sums_known_rates_and_respects_the_delay():
    ft = pd.date_range("2022-01-01", "2022-03-01", freq="8h", tz=UTC)
    f = pd.Series(1e-4, index=ft)
    t = pd.DatetimeIndex([ts("2022-01-20"), ts("2022-02-10"), ts("2022-02-11")])
    out = basis.trailing_funding(f, t)
    assert np.isnan(out[0])  # the history does not cover 30 days yet
    assert out[1] == pytest.approx(1e-4 * 90 * 365 / 30)  # 90 settlements in 30 days: 3 * 365 * 1e-4 a year
    # the settlement at exactly t is known only 5 minutes later, so changing it changes nothing at t
    g = f.copy()
    g[ts("2022-02-11")] = 0.05
    assert basis.trailing_funding(g, t)[2] == pytest.approx(out[2])
    g[ts("2022-02-10 16:00")] = 0.05
    assert basis.trailing_funding(g, t)[2] > out[2] + 0.1


def _contracts():
    # Binance-like listing gap: B is listed three days before A delivers
    return [Contract("X_220325", ts("2021-12-24 03:00"), ts("2022-03-25 08:00")),
            Contract("X_220624", ts("2022-03-22 03:00"), ts("2022-06-24 08:00"))]


def _closes(index, contracts, f_over_p=1.01):
    P = pd.Series(100.0, index=index)
    Q = pd.DataFrame({c.symbol: np.where((index >= c.listed) & (index < c.delivery), 100.0 * f_over_p, np.nan)
                      for c in contracts}, index=index)
    return P, Q


def test_front_contract_rolls_seven_days_before_delivery_and_waits_for_the_listing():
    idx = pd.date_range("2022-03-10", "2022-04-10", freq="1h", tz=UTC)
    cs = _contracts()
    P, Q = _closes(idx, cs)
    fund = pd.Series(0.0, index=pd.date_range("2021-12-01", "2022-04-10", freq="8h", tz=UTC))
    f = basis.features(idx, P, Q, cs, fund)
    front = pd.Series(basis.fronts(f), index=f.index, dtype=object)
    assert front[ts("2022-03-18")] == "X_220325"  # 7 days 8 hours to delivery
    assert front[ts("2022-03-19")] is None  # 6 days 8 hours: rolled out, the next contract is not listed yet
    assert front[ts("2022-03-23")] is None  # listed 2022-03-22 03:00, warm-up of 24 h not over
    assert front[ts("2022-03-24")] == "X_220624"
    row = f.loc[ts("2022-03-18")]
    assert row["days"] == pytest.approx(7 + 8 / 24)
    assert row["basis"] == pytest.approx(0.01 * 365 / (7 + 8 / 24))


def _feat(spread, front):
    n = len(spread)
    return pd.DataFrame({"pos": np.arange(n), "front": front, "spread": spread, "basis": spread},
                        index=pd.date_range("2022-01-01", periods=n, freq="1D", tz=UTC))


def test_r6b_enters_beyond_theta_exits_on_the_other_sign_and_at_rolls():
    s = [0.01, 0.03, 0.01, -0.01, -0.03, -0.04, 0.06, 0.06, 0.06, np.nan]
    fr = ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"]
    state, contract = basis.states_r6b(_feat(s, fr), theta=0.02)
    #            0.01 0.03 0.01 -0.01 -0.03 -0.04 0.06 roll(0.06) 0.06 nan
    assert list(state) == [0, 1, 1, 0, -1, -1, 1, 1, 1, 0]
    assert contract == [None, "A", "A", None, "A", "A", "A", "B", "B", None]
    state, contract = basis.states_r6b(_feat([0.03, 0.01, 0.01], ["A", "B", "B"]), theta=0.02)
    assert list(state) == [1, 0, 0]  # the roll closes the position and 0.01 does not reopen it


def test_weights_only_on_changes_with_the_roll_closing_the_old_contract():
    idx = pd.date_range("2022-01-01", periods=10, freq="1h", tz=UTC)
    feat = pd.DataFrame({"pos": [1, 3, 5, 7]}, index=idx[[1, 3, 5, 7]])
    W = basis.weights(idx, feat, [1, 1, 1, 0], ["A", "A", "B", None], "P", ["A", "B"])
    row = W.loc[idx[1]]
    assert row["P"] == 0.5 and row["A"] == -0.5 and np.isnan(row["B"])
    assert W.loc[idx[3]].isna().all()  # nothing changed: hold
    assert W.loc[idx[5]].tolist() == [0.5, 0.0, -0.5]  # roll: perp re-sized, A closed, B opened
    row = W.loc[idx[7]]
    assert row["P"] == 0.0 and row["B"] == 0.0 and np.isnan(row["A"])  # flat: perp and B closed
    assert W.drop(idx[[1, 5, 7]]).isna().all().all()


def test_engine_pnl_of_a_hedged_short_basis_position():
    """Perpetual flat at 100, quarterly converging from 101 to 100.5: the short quarterly earns 0.5 * (1 - F1/F0)
    minus entry costs; the perpetual leg earns nothing (funding zero)."""
    idx = pd.date_range("2022-01-01 22:00", periods=30, freq="1h", tz=UTC)
    F = pd.Series(np.linspace(101.0, 100.5, len(idx)), index=idx)
    C = pd.DataFrame({"P": 100.0, "Q": F})
    W = pd.DataFrame(np.nan, index=idx, columns=["P", "Q"])
    W.iloc[1] = [0.5, -0.5]  # decided at the close of the 23:00 bar, executed at the 00:00 open
    cm = CostModel(taker_fee=0.0005, slippage_bps={"P": 2.0, "Q": 3.0})
    res = engine.run(C, C, C, C, W, cm)
    f0, f1 = F.iloc[2], F.iloc[-1]  # open of the execution bar (open = close here) and the last close
    expected = 1.0 - 0.5 * (0.0005 + 2e-4) - 0.5 * (0.0005 + 3e-4) + 0.5 * (1 - f1 / f0)
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert res.positions.iloc[-1]["P"] == pytest.approx(0.5)  # sized on the equity before this rebalance's costs
    assert res.costs_paid == pytest.approx(0.5 * (0.0005 + 2e-4) + 0.5 * (0.0005 + 3e-4))


def _venue_like(seed=0, days=120):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days * 24, freq="1h", tz=UTC)
    P = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))), index=idx)
    cs = [Contract("X_220325", ts("2021-12-24 03:00"), ts("2022-03-25 08:00")),
          Contract("X_220624", ts("2022-02-10 03:00"), ts("2022-06-24 08:00"))]
    b = 0.05 + 0.03 * np.sin(np.arange(len(idx)) / 300) + rng.normal(0, 0.01, len(idx))
    Q = {}
    for c in cs:
        D = ((c.delivery - idx) / pd.Timedelta(days=1)).to_numpy()
        live = (idx >= c.listed) & (idx < c.delivery)
        Q[c.symbol] = np.where(live, P.to_numpy() * (1 + b * D / 365), np.nan)
    ft = pd.date_range(ts("2021-11-01"), idx[-1], freq="8h")
    fund = pd.Series(0.03 / 1095 + rng.normal(0, 5e-5, len(ft)), index=ft)
    return idx, P, pd.DataFrame(Q, index=idx), cs, fund


def _perturb_after(idx, P, Q, fund, cut, seed):
    """New prices and funding after the cut, as an adversary would choose them."""
    rng = np.random.default_rng(seed)
    later = idx >= cut  # bars opening at or after the cut close after it; the decision bar (cut - 1h) is known
    P2 = P.copy()
    P2[later] = P[later] * np.exp(rng.normal(0, 0.05, later.sum()))
    Q2 = Q.copy()
    Q2[later] = Q[later] * np.exp(rng.normal(0, 0.05, (later.sum(), Q.shape[1])))
    f2 = fund.copy()
    late_f = fund.index + config.METRICS_DELAY > cut
    f2[late_f] = rng.normal(0, 1e-3, late_f.sum())
    return P2, Q2, f2


def test_round6_features_and_states_use_no_future_data():
    idx, P, Q, cs, fund = _venue_like()
    f_full = basis.features(idx, P, Q, cs, fund)
    rng = np.random.default_rng(7)
    cuts = list(rng.choice(f_full.index[5:], 12, replace=False))
    for k, cut in enumerate(cuts):
        P2, Q2, f2 = _perturb_after(idx, P, Q, fund, cut, 100 + k)
        g = basis.features(idx, P2, Q2, cs, f2)
        known = f_full.index <= cut
        pd.testing.assert_frame_equal(f_full[known], g[known])
        assert not f_full[~known]["basis"].equals(g[~known]["basis"])  # the perturbation did reach later rows
        for theta in (0.02, 0.05):
            s1, c1 = basis.states_r6b(f_full, theta)
            s2, c2 = basis.states_r6b(g, theta)
            n = int(known.sum())
            assert list(s1[:n]) == list(s2[:n]) and c1[:n] == c2[:n]


def test_round6_weights_use_no_future_data():
    idx, P, Q, cs, fund = _venue_like(seed=3)
    f_full = basis.features(idx, P, Q, cs, fund)
    s, c = basis.states_r6b(f_full, 0.02)
    W = basis.weights(idx, f_full, s, c, "P", list(Q.columns))
    for k, cut in enumerate(f_full.index[10::15]):
        P2, Q2, f2 = _perturb_after(idx, P, Q, fund, cut, 300 + k)
        g = basis.features(idx, P2, Q2, cs, f2)
        s2, c2 = basis.states_r6b(g, 0.02)
        W2 = basis.weights(idx, g, s2, c2, "P", list(Q.columns))
        upto = W.index < cut  # weights decided at or before the cut sit on bars that open before it
        pd.testing.assert_frame_equal(W[upto], W2[upto])
