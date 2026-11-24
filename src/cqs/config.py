"""Project-wide constants. Changing any of these changes research results: log it in the registry."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Root of the repository (src/cqs/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Downloaded and derived data. Never committed. Override with CQS_DATA_DIR.
DATA_DIR = Path(os.environ.get("CQS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"

# Experiment registry (committed, append-only).
REGISTRY_PATH = Path(os.environ.get("CQS_REGISTRY", REPO_ROOT / "research" / "registry.jsonl"))

# Research protocol (Stage 1 document). The holdout is locked until the final test.
DEV_START = pd.Timestamp("2020-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-06-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-11-23 23:59:59", tz="UTC")

# Point-in-time availability delays.
METRICS_DELAY = pd.Timedelta(minutes=5)
OFF_EXCHANGE_DAILY_DELAY = pd.Timedelta(days=1)

# Binance public archive.
ARCHIVE_BASE_URL = "https://data.binance.vision"
ARCHIVE_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
METRICS_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

INTERVAL_TO_TIMEDELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
