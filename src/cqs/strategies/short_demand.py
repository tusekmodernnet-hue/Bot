"""Round 11 (research/preregistration.md, "Runda 11"): short older liquid tokens after a day of extremely negative
funding, the price of shorting a perpetual (Miller 1977; the shorting premium in equities).

R11A  the funding of day t is the sum of the rates settled from 00:00 of day t up to, not including, 00:00 of day
      t + 1 (UTC): the cost of a short held through the day, whether the contract settles every 8, 4 or 1 hours. An
      event is a day t on which a contract that was in the liquid universe at the close of t - 1 (contract age at
      least 90 days, 30-day average quote volume at least 20M USDT, no BTC, index contracts, stablecoins, known
      renames or dated contracts) has a funding of day t at or below -X. The shorts, their limits, the R9A stop and
      the top-20 alt basket hedge are those of R10A (`pumps.event_shorts`, `supply.basket_weights`): at the close of
      t + 1 the contract is shorted if that day's quote volume is at least 20M USDT, -0.05 of equity per short, at
      most 20 at a time, one per contract, H days in position, no new short for H days after a stop.
"""

from __future__ import annotations

import pandas as pd

from cqs.data import store
from cqs.strategies import pumps, supply


def load_funding(symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Settlement time x contract frame of the stored (holdout-guarded) funding rates, and the contracts without any."""
    fund, missing = {}, []
    for s in symbols:
        try:
            fr = store.load_funding(s)
        except FileNotFoundError:
            missing.append(s)
            continue
        if fr.empty:
            missing.append(s)
            continue
        fund[s] = fr.set_index("funding_time")["rate"]
    return pd.DataFrame(fund).sort_index(), missing


def daily_funding(funding: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Day x contract sum of the rates settled in [t 00:00, t + 1 00:00). NaN on a day without a settlement."""
    if funding.empty:
        return pd.DataFrame(index=index, columns=funding.columns, dtype=float)
    day = funding.index.floor("1D")
    return funding.groupby(day).sum(min_count=1).reindex(index)


def funding_events(funding: pd.DataFrame, universe: pd.DataFrame, threshold: float,
                   index: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows (contract, d0): funding of day d0 <= -threshold and a universe member at the close of d0 - 1.
    Both are known at the close of d0 + 1, the first decision of `pumps.event_shorts` with delay 1."""
    D = daily_funding(funding, index)
    member = universe.reindex(index=index, columns=D.columns, fill_value=False).fillna(False).astype(bool)
    hit = (D <= -threshold) & member.shift(1, fill_value=False)
    rows = [(c, t) for c in hit.columns for t in hit.index[hit[c].to_numpy()]]
    df = pd.DataFrame(rows, columns=["contract", "d0"])
    return df.sort_values(["d0", "contract"], ignore_index=True)


def r11a(open_, close, quote_volume, events, basket, hold, stop_k: float | None = pumps.STOP_K,
         size: float = pumps.SIZE, max_shorts: int = pumps.MAX_SHORTS, delay: int = pumps.DELAY):
    held, stops = pumps.event_shorts(open_, close, quote_volume, events, delay, hold, max_shorts, stop_k=stop_k)
    return supply.basket_weights(held, basket, close, size), held, stops
