"""Hyperliquid client on recorded-shape responses (no network): parsing, the holdout cut, funding pagination, the
survivorship check and a full download into a temporary store."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cqs import config
from cqs.data import guard, hyperliquid as HL
from cqs.data.archive import ArchiveError

MS = 1000
DAY = 86_400 * MS
H = 3_600 * MS
T0 = int(pd.Timestamp("2025-05-28", tz="UTC").timestamp() * MS)


def candle(t, o, h, l, c, v, n=10):
    return {"t": t, "T": t + DAY - 1, "s": "X", "i": "1d", "o": str(o), "h": str(h), "l": str(l), "c": str(c),
            "v": str(v), "n": n}


def test_candles_are_parsed_and_the_holdout_is_cut():
    rows = [candle(T0 + k * DAY, 10, 12, 8, 10, 100) for k in range(6)]          # 28.05 .. 02.06
    df = HL.candles_frame(rows)
    assert list(df["open_time"].dt.day) == [28, 29, 30, 31]                     # June is holdout
    assert df["quote_volume"].iloc[0] == pytest.approx(100 * (10 + 12 + 8 + 10) / 4)
    assert df["trades"].dtype.kind == "i" and df["close"].iloc[0] == 10.0


def test_end_of_requests_is_just_before_the_holdout():
    assert HL.end_ms() == int(config.HOLDOUT_START.timestamp() * MS) - 1
    with pytest.raises(guard.HoldoutAccessError):
        HL.end_ms("2025-06-02")


def test_funding_pages_until_an_empty_page(monkeypatch):
    calls = []
    pages = [[{"coin": "X", "fundingRate": "0.0001", "premium": "0.0", "time": T0 + k * H + 76} for k in range(3)],
             [{"coin": "X", "fundingRate": "-0.0002", "premium": "0.0", "time": T0 + k * H + 76} for k in (3, 4)],
             []]

    def fake(body, sources=None):
        calls.append(body)
        return pages[len(calls) - 1]
    monkeypatch.setattr(HL, "_post", fake)
    df = HL.fetch_funding("X", pd.Timestamp(T0, unit="ms", tz="UTC"))
    assert len(df) == 5 and df["rate"].iloc[-1] == pytest.approx(-0.0002)
    assert calls[1]["startTime"] == T0 + 2 * H + 76 + 1 and len(calls) == 3
    assert all(c["endTime"] == HL.end_ms() for c in calls)


def test_a_full_candle_page_is_an_error(monkeypatch):
    monkeypatch.setattr(HL, "_post", lambda body, sources=None: [candle(T0 - k * DAY, 1, 1, 1, 1, 1)
                                                                  for k in range(HL.MAX_CANDLES)])
    with pytest.raises(ArchiveError, match="history may be cut"):
        HL.fetch_candles("X")


def test_missing_check_counts_gaps_before_the_last_coin_with_data():
    meta = [{"asset": i, "name": n, "is_delisted": d} for i, (n, d) in
            enumerate([("BTC", False), ("OLD", True), ("A", False), ("GONE", True), ("B", False), ("NEW", False)])]
    out = HL.missing_check(meta, {"BTC", "OLD", "A", "B"})
    assert out["missing"] == ["GONE"] and out["listed_before_last_with_candles"] == 4
    assert out["missing_share"] == pytest.approx(0.25) and out["delisted_with_candles"] == 1


def test_download_writes_store_manifests_and_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PARQUET_DIR", tmp_path / "parquet")

    def fake(body, sources=None):
        if sources is not None:
            sources[json.dumps(body, sort_keys=True)] = "hash"
        if body["type"] == "meta":
            return {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
                                 {"name": "LATE", "szDecimals": 0, "maxLeverage": 3}]}
        if body["type"] == "candleSnapshot":
            if body["req"]["coin"] == "LATE":
                return []                                    # listed after the research period
            return [candle(T0 + k * DAY, 10, 11, 9, 10, 5) for k in range(4)]
        if body["type"] == "fundingHistory":
            return [] if body["startTime"] > T0 else [{"coin": "BTC", "fundingRate": "0.00001", "premium": "0",
                                                       "time": T0 + 5}]
        raise AssertionError(body)
    monkeypatch.setattr(HL, "_post", fake)
    s = HL.download(log=lambda m: None)
    assert s["with_research_candles"] == 1 and s["missing"] == [] and not s["failed"]
    k = HL.load_candles("BTC")
    assert len(k) == 4 and k["open_time"].max() < config.HOLDOUT_START
    assert len(HL.load_funding("BTC")) == 1
    man = json.loads(HL.store.parquet_path(HL.MARKET, HL.store.KLINES, "BTC", "1d").with_suffix(".manifest.json")
                     .read_text())
    assert man["rows"] == 4 and man["sources"]
    assert json.loads(HL.summary_path().read_text())["coins_in_meta"] == 2
