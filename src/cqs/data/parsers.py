"""Parsers for Binance archive files.

The archive format is not constant over time:
- older CSVs have no header row, newer ones do;
- spot files switched timestamps from milliseconds to microseconds in 2025.
Both are detected per file, never assumed.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cqs import config


def read_zip_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read the single CSV inside an archive zip, with or without a header row."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one CSV, found {names}")
        raw = zf.read(names[0])
    if not raw.strip():
        return pd.DataFrame(columns=columns)
    first_line = raw.split(b"\n", 1)[0].decode("utf-8").strip()
    has_header = first_line[:1].isalpha()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if df.shape[1] != len(columns):
        raise ValueError(f"{path}: expected {len(columns)} columns, found {df.shape[1]}")
    df.columns = columns
    return df


def epoch_to_utc(values: pd.Series) -> pd.Series:
    """Convert epoch timestamps to UTC datetimes, detecting ms vs us per value."""
    v = pd.to_numeric(values, errors="raise").astype("int64")
    # 1e14 in ms is year 5138; in us it is 1973. Anything above is microseconds.
    us = v > 10**14
    ms_values = np.where(us, v // 1000, v)
    return pd.to_datetime(ms_values, unit="ms", utc=True)


def parse_klines(path: Path) -> pd.DataFrame:
    df = read_zip_csv(path, config.KLINE_COLUMNS)
    out = pd.DataFrame(
        {
            "open_time": epoch_to_utc(df["open_time"]),
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["volume"].astype("float64"),
            "quote_volume": df["quote_volume"].astype("float64"),
            "trades": df["count"].astype("int64"),
            "taker_buy_volume": df["taker_buy_volume"].astype("float64"),
            "taker_buy_quote_volume": df["taker_buy_quote_volume"].astype("float64"),
        }
    )
    return out


def parse_funding(path: Path) -> pd.DataFrame:
    df = read_zip_csv(path, config.FUNDING_COLUMNS)
    # calc_time can carry a few milliseconds of jitter around the scheduled settlement; round to the minute.
    return pd.DataFrame(
        {
            "funding_time": epoch_to_utc(df["calc_time"]).round("min"),
            "interval_hours": df["funding_interval_hours"].astype("int64"),
            "rate": df["last_funding_rate"].astype("float64"),
        }
    )


def parse_metrics(path: Path) -> pd.DataFrame:
    df = read_zip_csv(path, config.METRICS_COLUMNS)
    out = df.drop(columns=["symbol"]).copy()
    out["create_time"] = pd.to_datetime(out["create_time"], utc=True)
    for c in out.columns:
        if c != "create_time":
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    # Point-in-time: a snapshot stamped t is usable from t + METRICS_DELAY.
    out["available_time"] = out["create_time"] + config.METRICS_DELAY
    return out
