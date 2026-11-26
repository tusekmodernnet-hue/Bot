"""Round 8 (research/preregistration.md, "Runda 8"): the supply mechanism behind R7C, three ways.

R8A  R7C's new-token shorts, hedged with an equal-weight long basket of the point-in-time top 20 without BTC
     instead of long BTC.
R8B  the same shorts after a new perpetual on a token that already traded on Binance spot at least 30 days before
     (the listings R7C leaves out as old tokens; Miller 1977: easier short selling), with the same basket hedge.
R8C  token age: short tokens 90 to A days after their first Binance listing (perpetual or spot), long tokens over
     730 days old, dollar neutral, membership fixed at each Sunday close.
All weights are targets for the engine; round 8 rebalances with a band of 20% of each target (for a -0.1 short
that is R7C's 2% of equity), and entries and exits always trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.strategies import listings

SIZE = listings.SIZE            # -0.1 of equity per event short, as in R7C
CONSISTENCY_TOL = 0.10          # perp / (multiplier x spot) at listing must be within 10% of 1
OLD_MIN_DAYS = 730
MIN_NAMES = 3
YOUNG_CAP = 0.1
YOUNG_LEG = 0.5
EXCLUDED = listings.INDEX | listings.STABLE | listings.RENAMES


def listing_consistent(perp_close: pd.Series, spot_close: pd.Series, mult: float, d0,
                       tol: float = CONSISTENCY_TOL) -> bool:
    """Median of perp / (mult x spot) over days D0 and D0 + 1 is within `tol` of 1. Both days are at or before
    every R8B decision (D0 + d, d >= 1). No common day means the pair cannot be confirmed."""
    d0 = pd.Timestamp(d0)
    pc = perp_close.loc[d0:d0 + pd.Timedelta(days=1)]
    r = (pc / (mult * spot_close.reindex(pc.index))).replace([np.inf, -np.inf], np.nan).dropna()
    return bool(len(r)) and abs(float(r.median()) - 1.0) <= tol


def old_token_events(first_perp: dict, first_spot: dict, consistent: dict, after: pd.Timestamp,
                     until: pd.Timestamp) -> pd.Series:
    """perp symbol -> D0 for a new perpetual (after < D0 <= until) on a token whose Binance spot pair first traded
    more than 30 days before D0, with agreeing prices at listing. Same exclusions as R7C."""
    ev = {}
    for p, d0 in first_perp.items():
        if d0 is None or p in EXCLUDED or p == listings.HEDGE:
            continue
        if not (after < d0 <= until):
            continue
        sp = first_spot.get(p)
        if sp is None or not sp < d0 - listings.SPOT_LEAD:
            continue
        if not consistent.get(p, False):
            continue
        ev[p] = pd.Timestamp(d0)
    return pd.Series(ev, dtype=object).sort_values() if ev else pd.Series(dtype=object)


def basket_weights(held: pd.DataFrame, basket: pd.DataFrame, close: pd.DataFrame,
                   size: float = SIZE) -> pd.DataFrame:
    """-size per held short plus an equal-weight long basket with the same total notional (dollar neutral).

    `basket` is a bool day x contract frame of point-in-time members. A member is left out on a day it is shorted
    or has no close. With no member left, the shorts are not held that day either (no unhedged shorts).
    """
    idx = held.index
    cols = list(dict.fromkeys(list(held.columns) + list(basket.columns)))
    S = -held.reindex(columns=cols, fill_value=False).astype(float) * size
    live = close.reindex(index=idx, columns=cols).notna()
    B = basket.reindex(index=idx, columns=cols, fill_value=False).fillna(False).astype(bool) & live & (S == 0)
    n = B.sum(axis=1)
    total = -S.sum(axis=1)
    S = S.mul((n > 0).astype(float), axis=0)
    per = (total / n.where(n > 0)).fillna(0.0)
    return S + B.astype(float).mul(per, axis=0)


def r8_event_shorts(close, quote_volume, events, basket, delay, hold):
    held = listings.shorts(close, quote_volume, events, delay, hold)
    return basket_weights(held, basket, close), held


# ------------------------------------------------------------------ R8C token age
def token_birth(first_perp: dict, first_spot: dict) -> dict:
    """First Binance listing day of each token: the earlier of its perpetual's and its spot pair's first day."""
    out = {}
    for p, d0 in first_perp.items():
        c = [d for d in (d0, first_spot.get(p)) if d is not None]
        out[p] = min(c) if c else None
    return out


def age_days(index: pd.DatetimeIndex, birth: dict) -> pd.DataFrame:
    return pd.DataFrame({s: (index - b).days for s, b in birth.items() if b is not None}, index=index)


def _weekly(m: pd.DataFrame) -> pd.DataFrame:
    """Membership decided at each Sunday close, kept until the next one (False before the first Sunday)."""
    s = m[m.index.dayofweek == 6].astype(float)
    return s.reindex(m.index).ffill().fillna(0.0).astype(bool)


def age_legs(universe: pd.DataFrame, age: pd.DataFrame, young_max: int,
             old_min: int = OLD_MIN_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in universe.columns if c in age.columns and c not in EXCLUDED and c != listings.HEDGE]
    U = universe[cols].fillna(False).astype(bool)
    A = age.reindex(index=U.index, columns=cols)
    return _weekly(U & (A <= young_max)), _weekly(U & (A > old_min))


def age_weights(young: pd.DataFrame, old: pd.DataFrame, close: pd.DataFrame, cap: float = YOUNG_CAP,
                leg: float = YOUNG_LEG, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Young names -cap each up to a -leg total (equal weights beyond leg / cap names), old names long in equal
    weights for the same total. A name without a bar that day is left out and the rest of its leg shares the
    notional. No position unless both legs have at least `min_names` names with a bar."""
    C = close.reindex(index=young.index, columns=young.columns)
    live = C.notna()
    y, o = young & live, old.reindex(index=young.index, columns=young.columns, fill_value=False) & live
    ny, no = y.sum(axis=1), o.sum(axis=1)
    on = ((ny >= min_names) & (no >= min_names)).astype(float)
    wy = np.minimum(cap, leg / ny.where(ny > 0)).fillna(0.0) * on
    wo = (wy * ny / no.where(no > 0)).fillna(0.0)
    return o.astype(float).mul(wo, axis=0) - y.astype(float).mul(wy, axis=0)


def r8c(universe, age, close, young_max):
    young, old = age_legs(universe, age, young_max)
    return age_weights(young, old, close), (young, old)
