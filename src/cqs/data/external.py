"""Off-exchange daily data: Fear & Greed (alternative.me), MVRV (Coin Metrics Community), FRED series (VIX,
3-month T-bill) and Deribit DVOL (30-day implied volatility index).

Every series is returned with the date on which it may be used for a decision: the observation for day d is
usable at the close of day d+1 (OFF_EXCHANGE_DAILY_DELAY). Coin Metrics values can be revised afterwards,
so they are not strictly point-in-time; treat any result that depends on them with extra suspicion.
"""

from __future__ import annotations

import io
import json

import pandas as pd

from cqs import config
from cqs.data.archive import _http_get

CACHE = lambda name: config.DATA_DIR / "external" / f"{name}.parquet"  # noqa: E731


def _cache(name: str, df: pd.Series) -> pd.Series:
    path = CACHE(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_frame("value").to_parquet(path)
    return df


def _usable(s: pd.Series) -> pd.Series:
    """Re-index an observation series by the day it becomes usable (obs day + delay)."""
    s = s.sort_index()
    s.index = s.index + config.OFF_EXCHANGE_DAILY_DELAY
    return s


def fear_greed() -> pd.Series:
    data = json.loads(_http_get("https://api.alternative.me/fng/?limit=0&format=json"))["data"]
    s = pd.Series({pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC").normalize(): float(d["value"]) for d in data})
    return _usable(_cache("fear_greed", s.sort_index()))


def coinmetrics(metric: str = "CapMVRVCur", asset: str = "btc") -> pd.Series:
    url = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"
           f"assets={asset}&metrics={metric}&frequency=1d&page_size=10000&start_time=2017-01-01")
    out = {}
    while url:
        page = json.loads(_http_get(url))
        for row in page["data"]:
            if row.get(metric) is not None:
                out[pd.Timestamp(row["time"]).tz_convert("UTC").normalize()] = float(row[metric])
        url = page.get("next_page_url")
    return _usable(_cache(f"cm_{asset}_{metric}", pd.Series(out).sort_index()))


def fred(series_id: str = "VIXCLS") -> pd.Series:
    # FRED drops connections from unknown user agents; a generic one works
    raw = _http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", user_agent="curl/8.5.0")
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["date"]).dt.tz_localize("UTC")).dropna()
    return _usable(_cache(f"fred_{series_id}", s))


def deribit_dvol(currency: str = "BTC") -> pd.Series:
    """Daily close of Deribit's DVOL (annualised 30-day implied volatility, in percent), from 2021-03-24.

    The API pages backwards from end_timestamp; `continuation` is the next end_timestamp.
    """
    end, rows = int(pd.Timestamp("2100-01-01", tz="UTC").timestamp() * 1000), {}
    start = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
    while True:
        url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?"
               f"currency={currency}&start_timestamp={start}&end_timestamp={end}&resolution=1D")
        res = json.loads(_http_get(url))["result"]
        for ts, _o, _h, _l, c in res["data"]:
            rows[pd.Timestamp(int(ts), unit="ms", tz="UTC").normalize()] = float(c)
        nxt = res.get("continuation")
        if not res["data"] or not nxt or nxt >= end:
            break
        end = int(nxt)
    return _usable(_cache(f"deribit_dvol_{currency}", pd.Series(rows).sort_index()))
