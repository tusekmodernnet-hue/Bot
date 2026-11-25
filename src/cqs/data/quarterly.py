"""Binance USDⓈ-M quarterly delivery contracts (BTCUSDT_210326, ...): delivery time, listing and 1h bars.

Delivery is at 08:00 UTC on the date in the symbol. The archive keeps printing records for some contracts after
delivery (e.g. BTCUSDT_230929 until 2023-11-14); those are not trades, so bars are kept only while the contract
was alive (open time before delivery). Until 2023-08 Binance listed the next quarterly only 3-9 days before the
current one delivered; from then on two quarterlies trade at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from cqs import config
from cqs.data import store

DELIVERY_HOUR = 8
_SYMBOL = re.compile(r"^([A-Z0-9]+USDT)_(\d{6})$")


@dataclass(frozen=True)
class Contract:
    symbol: str
    listed: pd.Timestamp  # open time of the first bar
    delivery: pd.Timestamp


def delivery(symbol: str) -> pd.Timestamp:
    m = _SYMBOL.match(symbol)
    if not m:
        raise ValueError(f"not a quarterly contract symbol: {symbol}")
    return pd.Timestamp(f"20{m.group(2)}", tz="UTC") + pd.Timedelta(hours=DELIVERY_HOUR)


def stored_symbols(base: str, interval: str = "1h", market: str = "futures/um") -> list[str]:
    """Quarterly contracts of `base` (e.g. "BTCUSDT") present in the local store, by delivery date."""
    root = config.PARQUET_DIR / market.replace("/", "_") / store.KLINES
    syms = [p.name for p in root.glob(f"{base}_*") if _SYMBOL.match(p.name) and (p / f"{interval}.parquet").exists()]
    return sorted(syms, key=delivery)


def cut_at_delivery(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return bars[bars["open_time"] < delivery(symbol)].reset_index(drop=True)


def load_bars(symbol: str, interval: str = "1h", end=None, market: str = "futures/um") -> pd.DataFrame:
    """Holdout-guarded bars of one quarterly contract, ending at delivery."""
    k = store.load_klines(symbol, interval, end=end, market=market, strip_zombies=False)
    return cut_at_delivery(k, symbol)
