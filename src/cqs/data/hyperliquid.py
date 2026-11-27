"""Hyperliquid public info API: the perpetual universe (delisted contracts included), daily candles and hourly
funding. Used by round 13 (research/preregistration.md, "Runda 13") to replicate R9A on another venue.

- Every request is a POST of a JSON body to https://api.hyperliquid.xyz/info. `meta` lists every perpetual ever
  added, in the order it was added (the list position is the asset number); delisted ones carry "isDelisted": true.
- Candles: `candleSnapshot`, interval 1d, open time `t` in ms (UTC days); o, h, l, c are strings; v is the volume in
  coins and n the number of trades. Hyperliquid gives no quote volume, so `quote_volume` = v x (o + h + l + c) / 4
  (the approximation stated in the pre-registration). The API keeps the latest 5000 candles per coin and interval.
- Funding: `fundingHistory`, one row per hourly settlement (`fundingRate` per hour, `premium`), paginated by time.
  Positive funding is paid by longs to shorts, as on Binance.
- Only data before the holdout start is requested (endTime = HOLDOUT_START - 1 ms) and later rows are dropped, so
  the holdout stays closed for this venue too. Every raw response is hashed (SHA-256) into the manifest next to its
  Parquet file, as for Deribit, so `store.data_fingerprint` and the holdout guard apply unchanged.

Run: python -m cqs.data.hyperliquid [COIN ...]  -> data/parquet/hyperliquid/..., download_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from cqs import config
from cqs.data import guard, store
from cqs.data.archive import USER_AGENT, ArchiveError

URL = "https://api.hyperliquid.xyz/info"
MARKET = "hyperliquid"
PAUSE = 1.1          # seconds after each request: about 55 a minute, under the public limit (1200 weight a minute)
MAX_CANDLES = 5000   # the API's cap per response; a full page means history may be cut, which must not pass silently
RETRY_CODES = (429, 500, 502, 503, 504)


def end_ms(end=None) -> int:
    """Last millisecond that may be requested: just before the holdout unless a declared final test is open."""
    return int(guard.resolve_end(end).timestamp() * 1000)


def _post(body: dict, sources: dict | None = None, retries: int = 6, timeout: float = 60.0):
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json",
                                                                 "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            if sources is not None:
                sources[json.dumps(body, sort_keys=True)] = hashlib.sha256(raw).hexdigest()
            time.sleep(PAUSE)
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES:
                raise ArchiveError(f"POST {body} failed: HTTP {e.code}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(min(60.0, 2.0 ** (attempt + 1)))
    raise ArchiveError(f"POST failed after {retries} attempts: {body}: {last}")


def fetch_meta(sources: dict | None = None) -> list[dict]:
    """Every perpetual in the universe, with its asset number (position in the list)."""
    res = _post({"type": "meta"}, sources)
    out = []
    for i, a in enumerate(res["universe"]):
        out.append({"asset": i, "name": a["name"], "is_delisted": bool(a.get("isDelisted", False)),
                    "only_isolated": bool(a.get("onlyIsolated", False)), "max_leverage": a.get("maxLeverage"),
                    "sz_decimals": a.get("szDecimals")})
    return out


def candles_frame(rows: list[dict]) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    o, h, l, c, v = (df[k].astype(float) for k in ("o", "h", "l", "c", "v"))
    out = pd.DataFrame({"open_time": pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True),
                        "open": o, "high": h, "low": l, "close": c, "volume": v,
                        "quote_volume": v * (o + h + l + c) / 4.0, "trades": df["n"].astype("int64")})
    out = out[out["open_time"] < config.HOLDOUT_START] if not guard.holdout_open() else out
    return out.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)


def fetch_candles(coin: str, sources: dict | None = None, interval: str = "1d", end=None) -> pd.DataFrame:
    rows = _post({"type": "candleSnapshot",
                  "req": {"coin": coin, "interval": interval, "startTime": 0, "endTime": end_ms(end)}}, sources)
    if len(rows) >= MAX_CANDLES:
        raise ArchiveError(f"{coin}: {len(rows)} candles in one response; history may be cut")
    return candles_frame(rows)


def funding_frame(rows: list[dict]) -> pd.DataFrame:
    cols = ["funding_time", "rate", "premium"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"funding_time": pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True),
                        "rate": df["fundingRate"].astype(float), "premium": df["premium"].astype(float)})
    out = out[out["funding_time"] < config.HOLDOUT_START] if not guard.holdout_open() else out
    return out.sort_values("funding_time").drop_duplicates("funding_time", keep="last").reset_index(drop=True)


def fetch_funding(coin: str, start, sources: dict | None = None, end=None, max_pages: int = 1000) -> pd.DataFrame:
    """All hourly settlements from `start` to the end, page by page until an empty page or the end."""
    t, stop = int(pd.Timestamp(start).timestamp() * 1000), end_ms(end)
    rows: list[dict] = []
    for _ in range(max_pages):
        if t > stop:
            break
        page = _post({"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": stop}, sources)
        if not page:
            break
        rows.extend(page)
        last = max(int(r["time"]) for r in page)
        if last < t:
            break
        t = last + 1
    else:
        raise ArchiveError(f"{coin}: funding history did not end after {max_pages} pages")
    return funding_frame(rows)


def _write(df: pd.DataFrame, dataset: str, coin: str, interval: str | None, sources: dict, tcol: str) -> Path:
    out = store.parquet_path(MARKET, dataset, coin, interval)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    manifest = {"market": MARKET, "dataset": dataset, "symbol": coin, "interval": interval, "rows": len(df),
                "first": str(df[tcol].iloc[0]) if len(df) else None,
                "last": str(df[tcol].iloc[-1]) if len(df) else None, "sources": sources}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def load_candles(coin: str, end=None) -> pd.DataFrame:
    return store._load(store.parquet_path(MARKET, store.KLINES, coin, "1d"), "open_time", None,
                       end).reset_index(drop=True)


def load_funding(coin: str, end=None) -> pd.DataFrame:
    return store._load(store.parquet_path(MARKET, store.FUNDING, coin), "funding_time", None,
                       end).reset_index(drop=True)


def meta_path() -> Path:
    return config.PARQUET_DIR / MARKET / "meta.json"


def summary_path() -> Path:
    return config.PARQUET_DIR / MARKET / "download_summary.json"


def missing_check(meta: list[dict], with_candles: set[str]) -> dict:
    """Pre-registered survivorship check: a contract added before the last contract that has research-period
    candles (a lower asset number) but has none itself counts as missing data."""
    numbered = [a for a in meta if a["name"] in with_candles]
    top = max((a["asset"] for a in numbered), default=-1)
    earlier = [a for a in meta if a["asset"] < top]
    missing = [a["name"] for a in earlier if a["name"] not in with_candles]
    return {"listed_before_last_with_candles": len(earlier), "missing": missing,
            "missing_share": len(missing) / len(earlier) if earlier else 0.0,
            "delisted_flagged": sum(a["is_delisted"] for a in meta),
            "delisted_with_candles": sum(a["is_delisted"] for a in numbered)}


def download(coins: list[str] | None = None, skip_existing: bool = True, log=print) -> dict:
    """Meta, daily candles and funding for every perpetual (or `coins`), research period only."""
    sources: dict = {}
    meta = fetch_meta(sources)
    meta_path().parent.mkdir(parents=True, exist_ok=True)
    meta_path().write_text(json.dumps({"fetched": pd.Timestamp.now(tz="UTC").isoformat(), "universe": meta,
                                       "sources": sources}, indent=1))
    names = [a["name"] for a in meta] if not coins else list(coins)
    with_candles, failed = set(), {}
    for k, coin in enumerate(names):
        kp = store.parquet_path(MARKET, store.KLINES, coin, "1d")
        fp = store.parquet_path(MARKET, store.FUNDING, coin)
        try:
            if skip_existing and kp.exists() and fp.exists():
                if len(pd.read_parquet(kp, columns=["open_time"])):
                    with_candles.add(coin)
                continue
            src: dict = {}
            k_df = fetch_candles(coin, src)
            _write(k_df, store.KLINES, coin, "1d", src, "open_time")
            if k_df.empty:
                continue
            with_candles.add(coin)
            src_f: dict = {}
            f_df = fetch_funding(coin, k_df["open_time"].iloc[0], src_f)
            _write(f_df, store.FUNDING, coin, None, src_f, "funding_time")
            log(f"{k + 1}/{len(names)} {coin}: {len(k_df)} days from {k_df['open_time'].iloc[0].date()}, "
                f"{len(f_df)} funding rows")
        except ArchiveError as e:
            failed[coin] = str(e)
            log(f"{k + 1}/{len(names)} {coin}: FAILED {e}")
    summ = {"coins_in_meta": len(meta), "requested": len(names), "with_research_candles": len(with_candles),
            "failed": failed, "end": str(guard.resolve_end(None))} | missing_check(meta, with_candles)
    summary_path().write_text(json.dumps(summ, indent=1))
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("coins", nargs="*", help="default: every perpetual in meta")
    ap.add_argument("--refresh", action="store_true", help="download again even if stored")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    s = download(a.coins or None, skip_existing=not a.refresh, log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in s.items() if k != "missing"}, indent=1), flush=True)
    return 0 if not s["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
