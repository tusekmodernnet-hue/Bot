"""Daily research panel: OHLC, volume, funding and the point-in-time universe, holdout-guarded."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cqs.data import store
from cqs.data.universe import UniverseRules, eligible


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    quote_volume: pd.DataFrame
    funding: pd.DataFrame  # settlement time x symbol (raw rates)
    universe: pd.DataFrame  # bool, day x symbol, decided at that day's close

    @property
    def returns(self) -> pd.DataFrame:
        """Close-to-close daily returns (NaN where not trading)."""
        return self.close.pct_change(fill_method=None)

    def daily_funding(self) -> pd.DataFrame:
        """Sum of funding rates settled within each UTC day (settlement in (day, day+1]), aligned to the
        day on which they are known: the day whose close is at or after the settlement."""
        f = self.funding.copy()
        day = (f.index - pd.Timedelta(microseconds=1)).normalize()  # 00:00 settlement belongs to previous day
        return f.groupby(day).sum(min_count=1).reindex(self.close.index)


def load(symbols: list[str], start=None, end=None, rules: UniverseRules = UniverseRules(),
         with_funding: bool = True) -> Panel:
    cols = {k: {} for k in ("open", "high", "low", "close", "quote_volume", "volume")}
    fund = {}
    for s in symbols:
        try:
            k = store.load_klines(s, "1d", start=start, end=end)
        except FileNotFoundError:
            continue
        if k.empty:
            continue
        k = k.set_index("open_time")
        for c in cols:
            cols[c][s] = k[c]
        if with_funding:
            try:
                fr = store.load_funding(s, start=start, end=end)
                fund[s] = fr.set_index("funding_time")["rate"]
            except FileNotFoundError:
                pass
    frames = {c: pd.DataFrame(v).sort_index() for c, v in cols.items()}
    uni = eligible({"close": frames["close"], "quote_volume": frames["quote_volume"], "volume": frames["volume"]}, rules)
    return Panel(frames["open"], frames["high"], frames["low"], frames["close"], frames["quote_volume"],
                 pd.DataFrame(fund).sort_index(), uni.reindex(columns=frames["close"].columns, fill_value=False))
