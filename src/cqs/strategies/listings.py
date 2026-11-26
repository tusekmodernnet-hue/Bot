"""R7C (research/preregistration.md, "Runda 7"): short new tokens after their perpetual listing, dollar neutral.

An event is the first traded day D0 of a USDT perpetual whose token is new to Binance: its USDT spot pair does not
exist or first traded less than 30 days before D0. Index contracts, stablecoins and known renames or
redenominations of existing tokens are excluded. At the close of day D0 + d the contract is shorted (executed at the
next open) if its average quote volume over days D0+1 .. D0+d is at least 20M USDT; the short is held for H days in
position or until the contract stops trading. Each short is -0.1 of equity, at most 10 at a time (an event that finds
every slot taken is skipped; among events of the same day the larger volume goes first), and BTC is held long for
the sum of the shorts. Weights are targets, so the engine's band trims a short that has grown.
The 90-day age rule of the universe does not apply here: new contracts are the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDEX = frozenset({"BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT"})
STABLE = frozenset({"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT"})
RENAMES = frozenset({"POLUSDT", "SUSDT", "RENDERUSDT", "KAIAUSDT", "BEAMXUSDT", "VICUSDT", "GUSDT", "EGLDUSDT",
                     "AAVEUSDT", "1000LUNCUSDT", "1000XECUSDT", "XECUSDT", "1000BTTCUSDT", "HIFIUSDT", "STRAXUSDT"})
SPOT_LEAD = pd.Timedelta(days=30)
MIN_QUOTE_VOLUME = 20e6
SIZE = 0.1
MAX_SHORTS = 10
HEDGE = "BTCUSDT"


def new_token_events(first_perp: dict, first_spot: dict, after: pd.Timestamp, until: pd.Timestamp) -> pd.Series:
    """perp symbol -> D0 for every qualifying listing with after < D0 <= until."""
    ev = {}
    for p, d0 in first_perp.items():
        if d0 is None or p in INDEX or p in STABLE or p in RENAMES or p == HEDGE:
            continue
        if not (after < d0 <= until):
            continue
        sp = first_spot.get(p)
        if sp is not None and sp < d0 - SPOT_LEAD:
            continue  # the token already traded on Binance spot: not new
        ev[p] = pd.Timestamp(d0)
    return pd.Series(ev, dtype=object).sort_values() if ev else pd.Series(dtype=object)


def shorts(close: pd.DataFrame, quote_volume: pd.DataFrame, events: pd.Series, delay: int, hold: int,
           max_shorts: int = MAX_SHORTS, min_qv: float = MIN_QUOTE_VOLUME) -> pd.DataFrame:
    """Bool day x contract: shorts held after each decision."""
    idx = close.index
    row = {t: i for i, t in enumerate(idx)}
    cols = [p for p in events.index if p in close.columns]
    C = close.reindex(columns=cols)
    Q = quote_volume.reindex(index=idx, columns=cols)
    by_row: dict[int, list] = {}
    for p in cols:
        d0 = events[p]
        t = pd.Timestamp(d0) + pd.Timedelta(days=delay)
        if t in row:
            liq = Q[p].loc[pd.Timestamp(d0) + pd.Timedelta(days=1):t].mean()
            by_row.setdefault(row[t], []).append((liq, p))
    out = np.zeros((len(idx), len(cols)), dtype=bool)
    col = {p: j for j, p in enumerate(cols)}
    Cv = C.to_numpy(dtype=float)
    active: dict[str, int] = {}
    for i in range(len(idx)):
        for p, i0 in list(active.items()):
            if i - i0 >= hold or not np.isfinite(Cv[i, col[p]]):
                del active[p]
        for liq, p in sorted(by_row.get(i, []), key=lambda t: (-t[0] if np.isfinite(t[0]) else np.inf, t[1])):
            if len(active) >= max_shorts:
                break
            if not (liq >= min_qv) or not np.isfinite(Cv[i, col[p]]):
                continue
            active[p] = i
        for p in active:
            out[i, col[p]] = True
    return pd.DataFrame(out, index=idx, columns=cols)


def weights(held: pd.DataFrame, size: float = SIZE, hedge: str = HEDGE) -> pd.DataFrame:
    w = -held.astype(float) * size
    w[hedge] = held.sum(axis=1) * size
    return w


def r7c(close, quote_volume, events, delay, hold):
    h = shorts(close, quote_volume, events, delay, hold)
    return weights(h), h
