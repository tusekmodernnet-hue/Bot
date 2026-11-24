"""Bar-level portfolio backtest engine.

Timing contract (no look-ahead):
- `weights.iloc[i]` is decided at the CLOSE of bar i, using only data available then.
- It is executed at the OPEN of bar i+1, at that price plus fees and slippage.
- A NaN weight means "no new decision": the position is kept as it is (it drifts with the price).
  A weight of 0 means flat. `rebalance_band` skips trades smaller than that fraction of equity.
- `stops.iloc[i]` (optional) is the stop price for the position held during bar i+1. If the bar opens
  beyond the stop the fill is the open (gap), otherwise the stop price; slippage is added on top.
  After a stop the asset stays flat until the strategy changes its target weight for that asset.
- Funding with settlement time in (open[i-1], open[i]] is charged at open[i] on the position held before
  the rebalance at open[i]: longs pay positive rates, shorts receive them.
- An asset with no bar (NaN open/close) cannot be traded. An open position in it is closed at its last
  close with the forced-exit slippage (delisting or exchange outage).

Positions are tracked as signed notional, so weights drift with prices between rebalances and turnover is
the real traded notional, not the change in target weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cqs.backtest.costs import CostModel


@dataclass
class BacktestResult:
    equity: pd.Series  # equity at the close of each bar (indexed by bar open time)
    positions: pd.DataFrame  # signed notional held at the close of each bar
    trades: pd.DataFrame  # one row per closed position episode
    fills: pd.DataFrame  # one row per execution
    costs_paid: float
    funding_paid: float
    traded_notional: float
    bar: pd.Timedelta
    initial_equity: float = 1.0

    @property
    def returns(self) -> pd.Series:
        r = self.equity.pct_change()
        r.iloc[0] = self.equity.iloc[0] / self.initial_equity - 1.0
        return r


def _frame(x: pd.DataFrame | None, index, columns, fill=np.nan) -> np.ndarray:
    if x is None:
        return np.full((len(index), len(columns)), fill)
    return x.reindex(index=index, columns=columns).to_numpy(dtype="float64")


def run(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    weights: pd.DataFrame,
    costs: CostModel,
    stops: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    initial_equity: float = 1.0,
    rebalance_band: float = 0.0,
) -> BacktestResult:
    """Simulate target weights on bar data. All frames: index = bar open times (UTC), columns = assets."""
    index = close.index
    assets = list(close.columns)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("bar index must be strictly increasing")
    bar = pd.Timedelta(pd.Series(index).diff().median())
    O, H, L, C = (_frame(f, index, assets) for f in (open_, high, low, close))
    W = _frame(weights, index, assets)  # NaN = keep the current position
    S = _frame(stops, index, assets)
    tradable = ~(np.isnan(O) | np.isnan(C))
    decided = ~np.isnan(W)
    if np.any(decided & (W != 0) & ~tradable):
        # A weight on a bar without data is a strategy bug (it used information it could not trade on).
        bad = np.argwhere(decided & (W != 0) & ~tradable)[0]
        raise ValueError(f"non-zero weight on untradable bar: {index[bad[0]]} {assets[bad[1]]}")
    rates = np.array([costs.rate(a) for a in assets])
    forced = costs.forced_rate()

    # Funding: sum of rates per bar window (open[i-1], open[i]].
    F = np.zeros((len(index), len(assets)))
    if funding is not None and costs.charge_funding:
        fr = funding.reindex(columns=assets).fillna(0.0)
        pos = index.searchsorted(fr.index, side="left")  # first bar open >= funding time
        for k, i in enumerate(pos):
            if 0 < i < len(index):
                F[i] += fr.iloc[k].to_numpy()

    n_bars, n = C.shape
    E = initial_equity
    p = np.zeros(n)
    stopped_target = np.full(n, np.nan)  # target weight that was stopped out, per asset
    eq = np.empty(n_bars)
    pos_hist = np.zeros((n_bars, n))
    costs_paid = 0.0
    funding_paid = 0.0
    traded = 0.0
    fills: list[tuple] = []
    # open episode per asset: [entry_index, side, pnl, entry_equity]
    ep_entry = np.full(n, -1)
    ep_side = np.zeros(n)
    ep_pnl = np.zeros(n)
    ep_eq = np.zeros(n)
    trades: list[tuple] = []

    def close_episode(j: int, i: int, reason: str) -> None:
        if ep_entry[j] >= 0:
            trades.append((assets[j], "long" if ep_side[j] > 0 else "short", index[ep_entry[j]], index[i],
                           i - ep_entry[j], ep_pnl[j], ep_pnl[j] / ep_eq[j], reason))
        ep_entry[j] = -1
        ep_pnl[j] = 0.0

    for i in range(n_bars):
        if i > 0:
            # a) forced exits: position in an asset that has no bar now
            for j in np.flatnonzero((p != 0) & ~tradable[i]):
                c = abs(p[j]) * forced
                E -= c
                costs_paid += c
                traded += abs(p[j])
                ep_pnl[j] -= c
                fills.append((index[i], assets[j], -p[j], C[i - 1, j], c, "forced_exit"))
                p[j] = 0.0
                close_episode(j, i, "forced_exit")
            live = tradable[i]
            # b) gap from previous close to this open
            g = np.where(live & (p != 0), O[i] / np.where(np.isnan(C[i - 1]), O[i], C[i - 1]) - 1.0, 0.0)
            E += float(np.sum(p * g))
            ep_pnl += p * g
            p = p * (1.0 + g)
            # c) funding on the position held through the settlement
            f = p * F[i]
            E -= float(np.sum(f))
            funding_paid += float(np.sum(f))
            ep_pnl -= f
            # d) rebalance to the target decided at the previous close
            w = W[i - 1]
            hold = np.isnan(w)
            keep_flat = ~np.isnan(stopped_target) & (hold | (w == stopped_target))
            stopped_target[~keep_flat] = np.nan
            target = np.where(hold, p, np.nan_to_num(w) * E)
            target = np.where(keep_flat | ~live, 0.0, target)
            trade = target - p
            moved = np.abs(trade) > max(rebalance_band * E, 1e-12 * max(E, 1e-12))
            moved &= ~(hold & ~keep_flat)  # a hold never trades
            if moved.any():
                c_vec = np.abs(trade) * rates
                c = float(np.sum(c_vec[moved]))
                E -= c
                costs_paid += c
                traded += float(np.sum(np.abs(trade[moved])))
                for j in np.flatnonzero(moved):
                    old, new = p[j], target[j]
                    fills.append((index[i], assets[j], trade[j], O[i, j], c_vec[j], "signal"))
                    closing = old != 0 and (new == 0 or np.sign(new) != np.sign(old))
                    opening = new != 0 and (old == 0 or np.sign(new) != np.sign(old))
                    if closing and opening:
                        # flip: |trade| = |old| + |new|; each episode carries its own leg's cost
                        ep_pnl[j] -= abs(old) * rates[j]
                        close_episode(j, i, "signal")
                        ep_entry[j], ep_side[j], ep_pnl[j], ep_eq[j] = i, np.sign(new), -abs(new) * rates[j], E
                    elif closing:
                        ep_pnl[j] -= c_vec[j]
                        close_episode(j, i, "signal")
                    elif opening:
                        ep_entry[j], ep_side[j], ep_pnl[j], ep_eq[j] = i, np.sign(new), -c_vec[j], E
                    else:
                        ep_pnl[j] -= c_vec[j]
                p = np.where(moved, target, p)
            # e) intrabar: stops, then mark to close
            stop_lvl = S[i - 1]
            for j in np.flatnonzero((p != 0) & ~np.isnan(stop_lvl)):
                s = stop_lvl[j]
                hit = (p[j] > 0 and L[i, j] <= s) or (p[j] < 0 and H[i, j] >= s)
                if not hit:
                    continue
                fill = min(O[i, j], s) if p[j] > 0 else max(O[i, j], s)
                r = fill / O[i, j] - 1.0
                pnl = p[j] * r
                exit_notional = abs(p[j] * (1.0 + r))
                c = exit_notional * rates[j]
                E += pnl - c
                costs_paid += c
                traded += exit_notional
                ep_pnl[j] += pnl - c
                fills.append((index[i], assets[j], -p[j] * (1.0 + r), fill, c, "stop"))
                stopped_target[j] = W[i - 1, j]
                p[j] = 0.0
                close_episode(j, i, "stop")
            r = np.where(live & (p != 0), C[i] / O[i] - 1.0, 0.0)
            E += float(np.sum(p * r))
            ep_pnl += p * r
            p = p * (1.0 + r)
        if E <= 0:
            raise RuntimeError(f"equity wiped out at {index[i]}: the strategy's risk is unacceptable")
        eq[i] = E
        pos_hist[i] = p

    # positions still open at the end are reported as open episodes, marked to the last close
    for j in np.flatnonzero(ep_entry >= 0):
        close_episode(j, n_bars - 1, "end_of_data")

    trades_df = pd.DataFrame(trades, columns=["asset", "side", "entry_time", "exit_time", "bars", "pnl",
                                              "return_on_equity", "exit_reason"])
    fills_df = pd.DataFrame(fills, columns=["time", "asset", "notional", "price", "cost", "reason"])
    return BacktestResult(
        equity=pd.Series(eq, index=index, name="equity"),
        positions=pd.DataFrame(pos_hist, index=index, columns=assets),
        trades=trades_df,
        fills=fills_df,
        costs_paid=costs_paid,
        funding_paid=funding_paid,
        traded_notional=traded,
        bar=bar,
        initial_equity=initial_equity,
    )
