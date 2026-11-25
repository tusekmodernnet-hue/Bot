"""Deribit public API: hourly candles of the BTC and ETH perpetuals and quarterly futures, and hourly perpetual
funding. Used by round 6 as the pre-registered second venue (confirmation, not a trial).

- Deribit's BTC and ETH futures are inverse (USD notional, settled in the coin). Round 6 uses their USD prices and
  treats P&L as linear in USD, like USDⓈ-M contracts; the pre-registration states this approximation.
- Quarterlies deliver at 08:00 UTC on the last Friday of March, June, September and December (BTC-25MAR22). The API
  lists only recently expired instruments, so names are built from that rule; a name with no candles raises.
- Candles: `get_tradingview_chart_data`, tick = bar open time. Funding: `get_funding_rate_history`, one row per
  hour; `interest_1h` is the rate paid over the hour ending at the row's timestamp.
- Data is fetched only up to the holdout start and stored like the Binance store (Parquet + manifest with the
  SHA-256 of every raw response), so `store.data_fingerprint` and the holdout guard apply unchanged.
"""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from cqs import config
from cqs.data import store
from cqs.data.archive import _http_get
from cqs.data.quarterly import Contract

BASE = "https://www.deribit.com/api/v2/public/"
MARKET = "deribit"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DELIVERY_HOUR = 8
H = 3600 * 1000


def last_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, calendar.monthrange(year, month)[1])
    return d - dt.timedelta(days=(d.weekday() - 4) % 7)


def quarterly_name(coin: str, year: int, month: int) -> str:
    d = last_friday(year, month)
    return f"{coin}-{d.day}{MONTHS[month - 1]}{year % 100:02d}"


def delivery(name: str) -> pd.Timestamp:
    """BTC-25MAR22 -> 2022-03-25 08:00 UTC."""
    code = name.split("-")[1]
    day, mon, yy = int(code[:-5]), MONTHS.index(code[-5:-2]) + 1, int(code[-2:])
    return pd.Timestamp(year=2000 + yy, month=mon, day=day, hour=DELIVERY_HOUR, tz="UTC")


def quarterly_names(coin: str, first_delivery: str, last_delivery: str) -> list[str]:
    """Quarterly contract names with delivery in [first_delivery, last_delivery] (dates)."""
    lo, hi = pd.Timestamp(first_delivery, tz="UTC"), pd.Timestamp(last_delivery, tz="UTC") + pd.Timedelta(days=1)
    out = []
    for y in range(lo.year, hi.year + 1):
        for m in (3, 6, 9, 12):
            name = quarterly_name(coin, y, m)
            if lo <= delivery(name) < hi:
                out.append(name)
    return out


def _ms(t) -> int:
    return int(pd.Timestamp(t).timestamp() * 1000)


def _get(url: str, sources: dict) -> dict:
    raw = _http_get(url)
    sources[url] = hashlib.sha256(raw).hexdigest()
    time.sleep(0.05)  # stay far below the public rate limit
    return json.loads(raw)["result"]


def fetch_candles(instrument: str, start, end, sources: dict, chunk_hours: int = 4000) -> pd.DataFrame:
    rows = []
    t0, t_end = _ms(start), _ms(end)
    while t0 < t_end:
        t1 = min(t0 + chunk_hours * H, t_end)
        url = (f"{BASE}get_tradingview_chart_data?instrument_name={instrument}&start_timestamp={t0}"
               f"&end_timestamp={t1}&resolution=60")
        res = _get(url, sources)
        if res.get("status") == "ok":
            rows.append(pd.DataFrame({"open_time": pd.to_datetime(res["ticks"], unit="ms", utc=True),
                                      "open": res["open"], "high": res["high"], "low": res["low"],
                                      "close": res["close"], "volume": res["volume"]}))
        t0 = t1  # chunks overlap by one bar; duplicates are dropped below
    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    df = pd.concat(rows, ignore_index=True)
    return df.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)


def fetch_funding(instrument: str, start, end, sources: dict, chunk_days: int = 30) -> pd.DataFrame:
    rows = []
    t0, t_end = _ms(start), _ms(end)
    while t0 < t_end:
        t1 = min(t0 + chunk_days * 24 * H, t_end)
        url = (f"{BASE}get_funding_rate_history?instrument_name={instrument}&start_timestamp={t0}"
               f"&end_timestamp={t1}")
        rows.extend(_get(url, sources))
        t0 = t1
    df = pd.DataFrame(rows)
    df = pd.DataFrame({"funding_time": pd.to_datetime(df["timestamp"], unit="ms", utc=True),
                       "rate": df["interest_1h"].astype(float), "interest_8h": df["interest_8h"].astype(float)})
    return df.sort_values("funding_time").drop_duplicates("funding_time", keep="last").reset_index(drop=True)


def _write(df: pd.DataFrame, dataset: str, instrument: str, interval: str | None, sources: dict, tcol: str) -> Path:
    out = store.parquet_path(MARKET, dataset, instrument, interval)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    manifest = {"market": MARKET, "dataset": dataset, "symbol": instrument, "interval": interval, "rows": len(df),
                "first": str(df[tcol].iloc[0]) if len(df) else None, "last": str(df[tcol].iloc[-1]) if len(df) else None,
                "sources": sources}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def build_candles(instrument: str, start, end=None) -> Path:
    end = end if end is not None else config.HOLDOUT_START - pd.Timedelta(hours=1)
    sources: dict = {}
    df = fetch_candles(instrument, start, end, sources)
    if df.empty:
        raise FileNotFoundError(f"Deribit returned no candles for {instrument}")
    if instrument.split("-")[1] != "PERPETUAL":
        df = df[df["open_time"] < delivery(instrument)]
    return _write(df, store.KLINES, instrument, "1h", sources, "open_time")


def build_funding(instrument: str, start, end=None) -> Path:
    end = end if end is not None else config.HOLDOUT_START - pd.Timedelta(hours=1)
    sources: dict = {}
    return _write(fetch_funding(instrument, start, end, sources), store.FUNDING, instrument, None, sources,
                  "funding_time")


def load_candles(instrument: str, end=None) -> pd.DataFrame:
    return store._load(store.parquet_path(MARKET, store.KLINES, instrument, "1h"), "open_time", None,
                       end).reset_index(drop=True)


def load_funding(instrument: str, end=None) -> pd.DataFrame:
    return store._load(store.parquet_path(MARKET, store.FUNDING, instrument), "funding_time", None,
                       end).reset_index(drop=True)


def contract(name: str) -> Contract:
    """Listing = open time of the first stored candle with volume (before it the API may print flat bars)."""
    k = load_candles(name)
    traded = k[k["volume"] > 0]
    return Contract(name, pd.Timestamp(traded["open_time"].iloc[0]), delivery(name))
