"""H3b, H4 and H7 (research/preregistration.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.data import store
from cqs.research.panel import Panel
from cqs.strategies.families import finalize


def daily_oi(symbol: str, days: pd.DatetimeIndex) -> pd.Series:
    """Open interest known at each day's close (00:00 UTC of the next day), using available_time."""
    m = store.load_metrics(symbol)[["available_time", "sum_open_interest"]].dropna().sort_values("available_time")
    closes = pd.DataFrame({"t": days + pd.Timedelta(days=1)})
    j = pd.merge_asof(closes, m, left_on="t", right_on="available_time", direction="backward")
    return pd.Series(j["sum_open_interest"].to_numpy(), index=days)


def h3b_crowding_ts(panel: Panel, z: float, hold_days: int) -> pd.DataFrame:
    f = panel.daily_funding()
    fz = (f - f.rolling(30, min_periods=30).mean()) / f.rolling(30, min_periods=30).std()
    sig = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    for a in panel.close.columns:
        oi = daily_oi(a, panel.close.index)
        oi_up = (oi / oi.shift(3) - 1.0) > 0.10
        trig = pd.Series(np.where(oi_up & (fz[a] > z), -1.0, np.where(oi_up & (fz[a] < -z), 1.0, 0.0)),
                         index=sig.index)
        held = np.zeros(len(trig))
        left, side = 0, 0.0
        for i, t in enumerate(trig.to_numpy()):
            if t != 0:
                left, side = hold_days, t
            held[i] = side if left > 0 else 0.0
            left = max(left - 1, 0)
        sig[a] = held
    return finalize(sig, panel)


def h4_carry(panel: Panel, spot_close: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Long spot / short perp while 3-day average funding, annualised, exceeds the threshold (exit below half).

    Returns weights over perp + spot columns (spot columns are named '<SYMBOL>_SPOT').
    """
    f = panel.daily_funding()
    ann = f.rolling(3, min_periods=3).mean() * 365.0
    cols = list(panel.close.columns) + [f"{a}_SPOT" for a in panel.close.columns]
    w = pd.DataFrame(0.0, index=panel.close.index, columns=cols)
    per_leg = 0.5 / len(panel.close.columns)  # 1x gross per asset pair, split across assets
    for a in panel.close.columns:
        on = False
        state = np.zeros(len(ann))
        for i, v in enumerate(ann[a].to_numpy()):
            if np.isnan(v):
                on = False
            elif not on and v > threshold:
                on = True
            elif on and v < threshold / 2:
                on = False
            state[i] = 1.0 if on else 0.0
        both = panel.close[a].notna() & spot_close[a].reindex(panel.close.index).notna()
        st = pd.Series(state, index=w.index).where(both, 0.0)
        w[a] = -2 * per_leg * st
        w[f"{a}_SPOT"] = 2 * per_leg * st
    return w, cols


def gate(weights: pd.DataFrame, block: pd.Series) -> pd.DataFrame:
    """H7: flat on days the external gate is active (block known at that day's close)."""
    b = block.reindex(weights.index).fillna(False).astype(bool)
    return weights.mask(b, 0.0, axis=0)
