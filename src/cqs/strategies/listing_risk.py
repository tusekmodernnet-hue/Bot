"""Round 9 (research/preregistration.md, "Runda 9"): risk control for the shorts of new perpetuals.

R9A  one pool of the R7C events (new tokens) and the R8B events (new perpetuals of tokens with Binance spot history),
     with their definitions and exclusions. At the close of D0 + 1 the contract is shorted (executed at the next open)
     if its quote volume that day is at least 20M USDT: -0.05 of equity per short, at most 20 at a time (among events
     of the same day the larger volume goes first, and an event that finds every slot taken is skipped), hedged with
     the equal-weight top-20 alt basket of round 8 for the same notional. A short ends after 30 days in position, when
     the contract stops trading, or at a stop: the first daily close at or above (1 + k) times the open at which the
     short was entered. The engine covers at the next open, the event is never shorted again and its slot is free.
     k None means no stop, which is exactly `listings.shorts`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.strategies import listings, supply

SIZE = 0.05
MAX_SHORTS = 20
DELAY = 1
HOLD = 30


def pooled_events(*sets: pd.Series) -> pd.Series:
    """Union of event sets (contract -> D0). The sets must not share a contract."""
    seen: set = set()
    for s in sets:
        dup = seen & set(s.index)
        if dup:
            raise ValueError(f"event sets overlap: {sorted(dup)[:5]}")
        seen |= set(s.index)
    parts = [s for s in sets if len(s)]
    return pd.concat(parts).sort_values() if parts else pd.Series(dtype=object)


def shorts_with_stop(open_: pd.DataFrame, close: pd.DataFrame, quote_volume: pd.DataFrame, events: pd.Series,
                     delay: int, hold: int, max_shorts: int = MAX_SHORTS, min_qv: float = listings.MIN_QUOTE_VOLUME,
                     stop_k: float | None = None, liq_from: int = 1) -> tuple[pd.DataFrame, list[dict]]:
    """Bool day x contract of the shorts held after each decision, as in `listings.shorts`, plus a close-based stop.

    A short decided at the close of row i0 is entered at the open of row i0 + 1. From row i0 + 1 on, a close at or
    above (1 + stop_k) x that open ends it at that row, so the engine covers at the next open; the freed slot can take
    an event decided at the same close. Every input used at row i is known at the close of row i.
    The volume filter averages days D0 + liq_from .. D0 + delay (default: from the first full day after the listing
    day; round 12 decides at the close of D0 itself, so it uses liq_from = 0, the listing day's own volume).
    Returns the held frame and one record per stop.
    """
    idx = close.index
    row = {t: i for i, t in enumerate(idx)}
    cols = [p for p in events.index if p in close.columns]
    C = close.reindex(columns=cols)
    O = open_.reindex(index=idx, columns=cols)
    Q = quote_volume.reindex(index=idx, columns=cols)
    by_row: dict[int, list] = {}
    for p in cols:
        d0 = events[p]
        t = pd.Timestamp(d0) + pd.Timedelta(days=delay)
        if t in row:
            liq = Q[p].loc[pd.Timestamp(d0) + pd.Timedelta(days=liq_from):t].mean()
            by_row.setdefault(row[t], []).append((liq, p))
    out = np.zeros((len(idx), len(cols)), dtype=bool)
    col = {p: j for j, p in enumerate(cols)}
    Cv, Ov = C.to_numpy(dtype=float), O.to_numpy(dtype=float)
    active: dict[str, int] = {}
    stops: list[dict] = []
    for i in range(len(idx)):
        for p, i0 in list(active.items()):
            j = col[p]
            if i - i0 >= hold or not np.isfinite(Cv[i, j]):
                del active[p]
                continue
            if stop_k is not None and i > i0:
                entry = Ov[i0 + 1, j]
                if np.isfinite(entry) and entry > 0 and Cv[i, j] >= (1.0 + stop_k) * entry:
                    stops.append({"contract": p, "decided": idx[i0], "stopped": idx[i], "entry_open": float(entry),
                                  "stop_close": float(Cv[i, j]), "days_in_position": i - i0})
                    del active[p]
        for liq, p in sorted(by_row.get(i, []), key=lambda t: (-t[0] if np.isfinite(t[0]) else np.inf, t[1])):
            if len(active) >= max_shorts:
                break
            if not (liq >= min_qv) or not np.isfinite(Cv[i, col[p]]):
                continue
            active[p] = i
        for p in active:
            out[i, col[p]] = True
    return pd.DataFrame(out, index=idx, columns=cols), stops


def r9a(open_, close, quote_volume, events, basket, stop_k, size: float = SIZE, max_shorts: int = MAX_SHORTS,
        delay: int = DELAY, hold: int = HOLD, liq_from: int = 1, min_qv: float = listings.MIN_QUOTE_VOLUME):
    """`min_qv`: the entry volume filter (default 20M USDT, R9A; round 13 uses 2M USD on Hyperliquid)."""
    held, stops = shorts_with_stop(open_, close, quote_volume, events, delay, hold, max_shorts, min_qv=min_qv,
                                   stop_k=stop_k, liq_from=liq_from)
    return supply.basket_weights(held, basket, close, size), held, stops


def stop_stats(stops: list[dict], open_: pd.DataFrame) -> dict:
    """Diagnostics only (never used for a decision): how far above the entry the stopped shorts were covered."""
    if not stops:
        return {"stops": 0}
    exit_ratio = []
    for s in stops:
        o = open_[s["contract"]]
        nxt = o.loc[o.index > s["stopped"]].dropna()
        if len(nxt):
            exit_ratio.append(float(nxt.iloc[0]) / s["entry_open"])
    r = np.array(exit_ratio)
    return {"stops": len(stops), "exit_over_entry_median": float(np.median(r)) if len(r) else None,
            "exit_over_entry_max": float(r.max()) if len(r) else None,
            "days_in_position_median": float(np.median([s["days_in_position"] for s in stops]))}
