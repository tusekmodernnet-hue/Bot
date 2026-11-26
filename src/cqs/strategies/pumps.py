"""Round 10 (research/preregistration.md, "Runda 10"): short liquid tokens after an extreme one-day rise.

R10A  an event is a day t on which a contract that was in the liquid universe at the close of t - 1 (contract age at
      least 90 days, 30-day average quote volume at least 20M USDT, no BTC, index contracts, stablecoins or known
      renames) closed at least X above its previous close. At the close of t + 1 the contract is shorted (executed at
      the next open) if that day's quote volume is at least 20M USDT: -0.05 of equity per short, at most 20 at a
      time, one short per contract at a time (an event on a contract already shorted is skipped), among events of the
      same day the larger volume first, hedged with the equal-weight top-20 alt basket of round 8. A short ends after
      H days in position, when the contract stops trading, or at the R9A stop: the first daily close at or above
      2 x the open at which it was entered. A later rise of the same contract is a new event, except within H days
      after a stop on that contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.strategies import listings, supply

SIZE = 0.05
MAX_SHORTS = 20
DELAY = 1
STOP_K = 1.0


def pump_events(close: pd.DataFrame, universe: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rows (contract, d0): close-to-close return of day d0 >= threshold and a universe member at the close of d0 - 1.
    Both are known at the close of d0."""
    ret = close.pct_change(fill_method=None)
    member = universe.reindex(index=close.index, columns=close.columns, fill_value=False).fillna(False).astype(bool)
    hit = (ret >= threshold) & member.shift(1, fill_value=False)
    rows = [(c, t) for c in hit.columns for t in hit.index[hit[c].to_numpy()]]
    df = pd.DataFrame(rows, columns=["contract", "d0"])
    return df.sort_values(["d0", "contract"], ignore_index=True)


def event_shorts(open_: pd.DataFrame, close: pd.DataFrame, quote_volume: pd.DataFrame, events: pd.DataFrame,
                 delay: int, hold: int, max_shorts: int = MAX_SHORTS, min_qv: float = listings.MIN_QUOTE_VOLUME,
                 stop_k: float | None = STOP_K) -> tuple[pd.DataFrame, list[dict]]:
    """Bool day x contract of held shorts for events that may repeat per contract, with the close-based stop of
    `listing_risk.shorts_with_stop`. One short per contract at a time: an event whose contract is held is skipped,
    and after a stop the contract is not shorted again for `hold` days."""
    idx = close.index
    row = {t: i for i, t in enumerate(idx)}
    cols = sorted(set(events["contract"]) & set(close.columns))
    C = close.reindex(columns=cols)
    O = open_.reindex(index=idx, columns=cols)
    Q = quote_volume.reindex(index=idx, columns=cols)
    by_row: dict[int, list] = {}
    for p, d0 in zip(events["contract"], events["d0"]):
        if p not in C.columns:
            continue
        t = pd.Timestamp(d0) + pd.Timedelta(days=delay)
        if t in row:
            liq = Q[p].loc[pd.Timestamp(d0) + pd.Timedelta(days=1):t].mean()
            by_row.setdefault(row[t], []).append((liq, p))
    out = np.zeros((len(idx), len(cols)), dtype=bool)
    col = {p: j for j, p in enumerate(cols)}
    Cv, Ov = C.to_numpy(dtype=float), O.to_numpy(dtype=float)
    active: dict[str, int] = {}
    blocked: dict[str, int] = {}   # contract -> first row at which it may be shorted again after a stop
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
                    blocked[p] = i + hold
        for liq, p in sorted(by_row.get(i, []), key=lambda t: (-t[0] if np.isfinite(t[0]) else np.inf, t[1])):
            if len(active) >= max_shorts:
                break
            if p in active or i < blocked.get(p, -1) or not (liq >= min_qv) or not np.isfinite(Cv[i, col[p]]):
                continue
            active[p] = i
        for p in active:
            out[i, col[p]] = True
    return pd.DataFrame(out, index=idx, columns=cols), stops


def r10a(open_, close, quote_volume, events, basket, hold, stop_k: float | None = STOP_K, size: float = SIZE,
         max_shorts: int = MAX_SHORTS, delay: int = DELAY):
    held, stops = event_shorts(open_, close, quote_volume, events, delay, hold, max_shorts, stop_k=stop_k)
    return supply.basket_weights(held, basket, close, size), held, stops
