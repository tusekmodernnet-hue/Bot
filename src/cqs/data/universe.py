"""Point-in-time trading universe.

On each day the universe holds only contracts that a trader could have picked on that day, using only
information available at that day's close: listed long enough, actually trading, and liquid enough.
Contracts that were later delisted are included for as long as they traded, which is what removes
survivorship bias. Measured on 2026-11-24: of 165 USDT perpetuals listed by the end of 2022, 56 (34%)
no longer trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cqs.data import store


@dataclass(frozen=True)
class UniverseRules:
    min_age_days: int = 90
    liquidity_window_days: int = 30
    min_avg_quote_volume: float = 20e6  # USDT per day, trailing average
    top_n: int | None = 50
    quote_asset: str = "USDT"


def daily_panel(symbols: list[str], market: str = "futures/um", start=None, end=None) -> dict[str, pd.DataFrame]:
    """Wide date x symbol frames (close, quote_volume, volume) from stored 1d klines.

    A symbol is NaN on days it did not exist or no longer traded (zombie tail removed).
    """
    closes, qvols, vols = {}, {}, {}
    for s in symbols:
        try:
            k = store.load_klines(s, "1d", start=start, end=end, market=market)
        except FileNotFoundError:
            continue
        if k.empty:
            continue
        idx = k["open_time"].dt.normalize()
        closes[s] = pd.Series(k["close"].to_numpy(), index=idx)
        qvols[s] = pd.Series(k["quote_volume"].to_numpy(), index=idx)
        vols[s] = pd.Series(k["volume"].to_numpy(), index=idx)
    return {
        "close": pd.DataFrame(closes).sort_index(),
        "quote_volume": pd.DataFrame(qvols).sort_index(),
        "volume": pd.DataFrame(vols).sort_index(),
    }


def eligible(panel: dict[str, pd.DataFrame], rules: UniverseRules = UniverseRules()) -> pd.DataFrame:
    """Boolean date x symbol frame: True where the contract is in the universe at that day's close.

    Every input is a trailing window ending at the same day, so the frame uses no future information.
    """
    close = panel["close"]
    qv = panel["quote_volume"]
    vol = panel["volume"]
    cols = [c for c in close.columns if c.endswith(rules.quote_asset)]
    close, qv, vol = close[cols], qv[cols], vol[cols]
    exists = close.notna()
    age = exists.cumsum()  # days since first bar, counting only days with a bar
    trading = vol.fillna(0) > 0
    avg_qv = qv.rolling(rules.liquidity_window_days, min_periods=rules.liquidity_window_days).mean()
    ok = exists & trading & (age >= rules.min_age_days) & (avg_qv >= rules.min_avg_quote_volume)
    if rules.top_n is not None:
        rank = avg_qv.where(ok).rank(axis=1, ascending=False, method="first")
        ok = ok & (rank <= rules.top_n)
    return ok
