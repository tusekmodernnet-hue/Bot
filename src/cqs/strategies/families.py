"""Pre-registered strategy families (research/preregistration.md). Each returns daily target weights
decided at the close of each day: NaN = hold (no decision that day), 0 = flat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.portfolio import sizing
from cqs.research.panel import Panel

MIN_CROSS_SECTION = 10


def finalize(signal: pd.DataFrame, panel: Panel, rebalance_every: int = 1, legs_neutral: bool = False,
             beta_neutral: bool = False, vol_overlay: bool = True) -> pd.DataFrame:
    """Signal (+1/-1/0, NaN = no view) -> sized, capped weights on the rebalance schedule."""
    r = panel.returns
    sig = signal.where(panel.universe, 0.0).fillna(0.0)
    w = sizing.inverse_vol(sig, r).fillna(0.0)
    if legs_neutral:
        longs = w.clip(lower=0)
        shorts = w.clip(upper=0)
        lg = longs.sum(axis=1)
        sg = -shorts.sum(axis=1)
        both = (lg > 0) & (sg > 0)
        w = (longs.div(lg.where(both), axis=0) * 0.5 + shorts.div(sg.where(both), axis=0) * 0.5).fillna(0.0)
    else:
        w = sizing.normalize_gross(w).fillna(0.0)
    if beta_neutral:
        beta = sizing.beta_to(r).reindex_like(w)
        port_beta = (w * beta.fillna(1.0)).sum(axis=1)
        if "BTCUSDT" in panel.universe:
            w["BTCUSDT"] = w["BTCUSDT"] - port_beta.where(panel.universe["BTCUSDT"], 0.0)
    if vol_overlay:  # H5: without it the book stays at gross 1x
        w = sizing.vol_target(w, r)
    w = sizing.cap_positions(w).fillna(0.0)
    # never hold a weight on a day without data
    w = w.where(panel.close.notna(), 0.0)
    if rebalance_every > 1:
        keep = np.zeros(len(w), dtype=bool)
        keep[::rebalance_every] = True
        w = w.where(pd.Series(keep, index=w.index), np.nan, axis=0)
        # exits forced by the universe are executed immediately, not at the next rebalance
        w = w.where(~((~panel.universe) & panel.close.notna()), 0.0)
    return w


def _xs(score: pd.DataFrame, panel: Panel, q: float = 0.2, contrarian: bool = False) -> pd.DataFrame:
    """+1 for the top quantile of the score, -1 for the bottom (reversed if contrarian), within the universe."""
    s = score.where(panel.universe)
    n = s.notna().sum(axis=1)
    pct = s.rank(axis=1, pct=True)
    sig = (pct > 1 - q).astype(float) - (pct <= q).astype(float)
    sig = sig.where(s.notna(), 0.0).where(n >= MIN_CROSS_SECTION, 0.0)
    return -sig if contrarian else sig


# H1 — time-series momentum
def h1_tsmom(panel: Panel, lookback: int, long_only: bool, vol_overlay: bool = True) -> pd.DataFrame:
    ret = panel.close / panel.close.shift(lookback) - 1.0
    sig = np.sign(ret)
    if long_only:
        sig = sig.clip(lower=0)
    return finalize(sig, panel, vol_overlay=vol_overlay)


# H2a — cross-sectional momentum, skipping the last day
def h2a_xsmom(panel: Panel, lookback: int, beta_neutral: bool, vol_overlay: bool = True) -> pd.DataFrame:
    score = panel.close.shift(1) / panel.close.shift(lookback + 1) - 1.0
    return finalize(_xs(score, panel), panel, rebalance_every=7, legs_neutral=True, beta_neutral=beta_neutral,
                    vol_overlay=vol_overlay)


# H2b — short-term cross-sectional reversal
def h2b_reversal(panel: Panel, lookback: int) -> pd.DataFrame:
    score = panel.close / panel.close.shift(lookback) - 1.0
    return finalize(_xs(score, panel, contrarian=True), panel, legs_neutral=True)


# H3a — cross-sectional funding crowding: long the cheapest-to-hold longs, short the most crowded
def h3a_funding_xs(panel: Panel, window_days: int, vol_overlay: bool = True, rebalance_every: int = 1) -> pd.DataFrame:
    """H3a; R2B is the same ranking with rebalance_every in {7, 14}."""
    f = panel.daily_funding().rolling(window_days, min_periods=window_days).mean()
    return finalize(_xs(f, panel, contrarian=True), panel, rebalance_every=rebalance_every, legs_neutral=True,
                    vol_overlay=vol_overlay)


GRIDS = {
    "H1": [("h1_tsmom", dict(lookback=L, long_only=lo)) for L in (20, 40, 60, 90, 120) for lo in (False, True)],
    "H2a": [("h2a_xsmom", dict(lookback=L, beta_neutral=b)) for L in (7, 14, 30, 60, 90) for b in (False, True)],
    "H2b": [("h2b_reversal", dict(lookback=L)) for L in (1, 3)],
    "H3a": [("h3a_funding_xs", dict(window_days=W)) for W in (1, 3, 7, 14)],
}
FUNCS = {"h1_tsmom": h1_tsmom, "h2a_xsmom": h2a_xsmom, "h2b_reversal": h2b_reversal, "h3a_funding_xs": h3a_funding_xs}
