"""R7A (research/preregistration.md, "Runda 7"): spot-perpetual cash and carry on many coins.

Each held pair is long 1/K of equity in the spot coin and short 1/K in its perpetual, so the pair earns the funding
the short receives while price moves cancel (up to the spot-perpetual basis). A pair is opened when the 3-day funding
known at the decision, annualised, exceeds the T-bill yield by more than M, and closed when that excess falls below
M/2, when the pair stops being available, or when the contract fails the universe's age, trading and liquidity tests
(the top-50 rank is needed only to open). At most K pairs are held; free slots go to the highest excess, and a held
pair is never swapped for a better one. Exits are processed before entries on each decision.

Everything at row t uses data known at the close of day t (the engine executes at the next open).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.strategies import basis

FUNDING_DAYS = 3
SPOT_MIN_AGE = 30
SPOT = "_SPOT"


def funding_known(funding: pd.DataFrame, days: pd.DatetimeIndex, window_days: int = FUNDING_DAYS) -> pd.DataFrame:
    """Annualised funding over the `window_days` days before each daily decision (the close of day d is d + 1 day),
    counting only settlements known 5 minutes after they happen. NaN until the history covers the window."""
    times = days + pd.Timedelta(days=1)
    out = {s: basis.trailing_funding(funding[s], times, days=window_days) for s in funding.columns}
    return pd.DataFrame(out, index=days)


def pair_available(perp_close: pd.DataFrame, spot_close: pd.DataFrame, mult: dict[str, float],
                   min_age: int = SPOT_MIN_AGE, window: int = 7, tol: float = 0.10) -> pd.DataFrame:
    """True where both closes exist, the spot pair has traded on at least `min_age` days and the median of
    perp / (mult * spot) over the last `window` days with both prices is within `tol` of 1."""
    out = {}
    for s in perp_close.columns:
        if s not in spot_close.columns:
            out[s] = pd.Series(False, index=perp_close.index)
            continue
        p, q = perp_close[s], spot_close[s].reindex(perp_close.index)
        age = q.notna().cumsum()
        ratio = p / (mult.get(s, 1.0) * q)
        med = ratio.rolling(window, min_periods=window).median()
        out[s] = p.notna() & q.notna() & (age >= min_age) & ((med - 1.0).abs() <= tol)
    return pd.DataFrame(out, index=perp_close.index)


def held_pairs(excess: pd.DataFrame, can_open: pd.DataFrame, can_hold: pd.DataFrame, margin: float,
               slots: int) -> pd.DataFrame:
    """Bool day x coin: the pairs held after each decision."""
    cols = list(excess.columns)
    E = excess.to_numpy(dtype=float)
    O = can_open.reindex_like(excess).fillna(False).to_numpy(dtype=bool)
    H = can_hold.reindex_like(excess).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(E.shape, dtype=bool)
    held: set[int] = set()
    for i in range(E.shape[0]):
        e = E[i]
        for j in list(held):
            if not H[i, j] or not (e[j] >= margin / 2):  # a NaN excess also closes the pair
                held.discard(j)
        free = slots - len(held)
        if free > 0:
            cand = [j for j in range(len(cols)) if j not in held and O[i, j] and e[j] > margin]
            cand.sort(key=lambda j: (-e[j], cols[j]))
            held.update(cand[:free])
        for j in held:
            out[i, j] = True
    return pd.DataFrame(out, index=excess.index, columns=cols)


def weights(held: pd.DataFrame, slots: int) -> pd.DataFrame:
    """Target weights: -1/K on each held perpetual and +1/K on its spot leg ('<SYMBOL>_SPOT'), 0 elsewhere."""
    h = held.astype(float) / slots
    w = pd.concat([-h, h.rename(columns=lambda c: c + SPOT)], axis=1)
    return w


def r7a(excess: pd.DataFrame, can_open: pd.DataFrame, can_hold: pd.DataFrame, margin: float,
        slots: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = held_pairs(excess, can_open, can_hold, margin, slots)
    return weights(h, slots), h
