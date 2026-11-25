"""Limit-order simulation on 1-minute bars (round 4, research/preregistration.md "Runda 4").

Conservative fill rules:
- A buy limit fills only when a bar's low is strictly below the limit (price traded through the level); a sell
  limit when the high is strictly above. Fill at the limit price, maker fee, no slippage.
- Take profit is a limit order (same trade-through rule), active from the bar after the entry.
- Stop loss is a stop-market order: triggered when low <= stop (long) / high >= stop (short), filled at the stop
  or at a worse open (gap), plus slippage and the taker fee. In the entry bar the stop is filled at the stop.
- When take profit and stop both lie inside one bar, the stop is assumed. In the entry bar only the stop counts.
- A time exit is a market order at the open of the first bar after the deadline (taker fee and slippage).
- When both sides of a two-sided quote could fill in the same bar, the side with the worse result at that bar's
  close is assumed.

Orders come as "plans": two-sided (or one-sided) limit quotes active on bars [start, end], each side with its
take profit and stop, and a deadline for the resulting trade (absolute bar, or minutes after the fill). One plan
or one position at a time: plans that start while a position is open, or overlap the active plan, are skipped.
Size: `risk` of equity to the stop, at most `cap` x equity. Funding is charged on the notional held at each
settlement. Each contract is simulated on its own equity; a two-contract portfolio's daily return is the sum of
the contracts' daily returns (equivalent to sizing on the shared equity at the start of each day).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fees:
    maker: float = 0.0002
    taker: float = 0.0005
    slippage: float = 0.0002  # per side, on market (stop and time) exits

    def scaled(self, k: float) -> "Fees":
        return Fees(self.maker * k, self.taker * k, self.slippage * k)


@dataclass
class Plans:
    """One row per plan; NaN price = no order on that side; deadline_abs = -1 and deadline_rel = -1 = none."""
    start: np.ndarray
    end: np.ndarray
    buy_px: np.ndarray
    buy_tp: np.ndarray
    buy_sl: np.ndarray
    sell_px: np.ndarray
    sell_tp: np.ndarray
    sell_sl: np.ndarray
    deadline_abs: np.ndarray
    deadline_rel: np.ndarray

    def __len__(self) -> int:
        return len(self.start)


def simulate(index: pd.DatetimeIndex, o, h, l, c, plans: Plans, fees: Fees, funding: pd.Series | None = None,
             risk: float = 0.005, cap: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    """Equity at each bar's close and the list of trades."""
    n = len(index)
    o, h, l, c = (np.asarray(x, float).tolist() for x in (o, h, l, c))
    fund = [0.0] * n
    if funding is not None and len(funding):
        pos = index.searchsorted(funding.index, side="left")  # charged at the first bar open at/after settlement
        for k, i in enumerate(pos):
            if 0 < i < n:
                fund[i] += float(funding.iloc[k])
    P = {k: np.asarray(getattr(plans, k)).tolist() for k in plans.__dataclass_fields__}
    m = len(plans)
    eq = np.empty(n)
    E = 1.0
    side = 0  # +1 long, -1 short, 0 flat
    units = entry_px = tp = sl = 0.0
    dl = -1
    entry_i = -1
    entry_E = 0.0
    trades = []
    k = 0  # next plan
    act = -1  # active plan
    for i in range(n):
        # time exit at this open, funding on the position held into this open
        if side != 0:
            if fund[i] != 0.0:
                E -= side * units * o[i] * fund[i]
            if 0 <= dl < i:
                px = o[i] * (1 - side * fees.slippage)
                E += side * units * (px - c[i - 1]) - units * px * fees.taker
                trades.append((index[entry_i], index[i], side, entry_px, px, "time", E / entry_E - 1))
                side = 0
        # expire the active plan, then activate the plan that starts here (only when flat)
        if act != -1 and i > P["end"][act]:
            act = -1
        while k < m and P["start"][k] < i:
            k += 1
        if k < m and P["start"][k] == i:
            if side == 0 and act == -1:
                act = k
            k += 1
        if side == 0 and act != -1:
            fills = []
            bp, sp = P["buy_px"][act], P["sell_px"][act]
            if bp == bp and l[i] < bp:
                fills.append((1, bp, P["buy_tp"][act], P["buy_sl"][act]))
            if sp == sp and h[i] > sp:
                fills.append((-1, sp, P["sell_tp"][act], P["sell_sl"][act]))
            best = None
            for d, px, t_, s_ in fills:
                dist = (px - s_) / px * d
                if not dist > 0:
                    continue
                u = min(risk / dist, cap) * E / px
                stopped = (d > 0 and l[i] <= s_) or (d < 0 and h[i] >= s_)
                if stopped:
                    out_px = s_ * (1 - d * fees.slippage)
                    val = d * u * (out_px - px) - u * px * fees.maker - u * out_px * fees.taker
                else:
                    val = d * u * (c[i] - px) - u * px * fees.maker
                if best is None or val < best[0]:
                    best = (val, d, px, t_, s_, u, stopped)
            if best is not None:
                val, d, px, t_, s_, u, stopped = best
                entry_E, entry_i, entry_px = E, i, px
                E += val
                if stopped:
                    out_px = s_ * (1 - d * fees.slippage)
                    trades.append((index[i], index[i], d, px, out_px, "stop", E / entry_E - 1))
                else:
                    side, units, tp, sl = d, u, t_, s_
                    da, dr = P["deadline_abs"][act], P["deadline_rel"][act]
                    dl = da if da >= 0 else (i + dr if dr >= 0 else -1)
                act = -1  # a filled plan is done (the other side is cancelled)
                eq[i] = E
                continue
        if side != 0 and entry_i != i:
            prev = c[i - 1]
            if (side > 0 and l[i] <= sl) or (side < 0 and h[i] >= sl):
                raw = min(o[i], sl) if side > 0 else max(o[i], sl)
                px = raw * (1 - side * fees.slippage)
                E += side * units * (px - prev) - units * px * fees.taker
                trades.append((index[entry_i], index[i], side, entry_px, px, "stop", E / entry_E - 1))
                side = 0
            elif (side > 0 and h[i] > tp) or (side < 0 and l[i] < tp):
                E += side * units * (tp - prev) - units * tp * fees.maker
                trades.append((index[entry_i], index[i], side, entry_px, tp, "target", E / entry_E - 1))
                side = 0
            else:
                E += side * units * (c[i] - prev)
        if E <= 0:
            raise RuntimeError(f"equity wiped out at {index[i]}")
        eq[i] = E
    if side != 0:
        trades.append((index[entry_i], index[n - 1], side, entry_px, c[n - 1], "end_of_data", E / entry_E - 1))
    tdf = pd.DataFrame(trades, columns=["entry_time", "exit_time", "side", "entry", "exit", "reason", "return"])
    return pd.Series(eq, index=index, name="equity"), tdf
