"""Parquet store: build per-symbol datasets from the archive and load them point-in-time.

Each Parquet file has a sidecar manifest with the archive keys and SHA-256 of the source zips, so an
experiment can record exactly which data it used (`data_fingerprint`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from cqs import config
from cqs.data import archive, guard, parsers
from cqs.data.validate import strip_zombie_tail

KLINES = "klines"
FUNDING = "fundingRate"
METRICS = "metrics"

_PARSERS = {
    KLINES: parsers.parse_klines,
    "premiumIndexKlines": parsers.parse_klines,
    "markPriceKlines": parsers.parse_klines,
    "indexPriceKlines": parsers.parse_klines,
    FUNDING: parsers.parse_funding,
    METRICS: parsers.parse_metrics,
}
_TIME_COL = {FUNDING: "funding_time", METRICS: "create_time"}


def parquet_path(market: str, dataset: str, symbol: str, interval: str | None = None) -> Path:
    name = f"{interval}.parquet" if interval else "all.parquet"
    return config.PARQUET_DIR / market.replace("/", "_") / dataset / symbol / name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(market: str, dataset: str, symbol: str, interval: str | None = None, workers: int = 16) -> Path:
    """Download (cached) every archive file for one symbol and write a single sorted Parquet file."""
    spec = archive.DatasetSpec(market, dataset, interval)
    keys = archive.files_for_symbol(spec, symbol)
    if not keys:
        raise FileNotFoundError(f"no archive files for {market}/{dataset}/{symbol}/{interval}")
    paths = archive.download_many(keys, workers=workers)
    parse = _PARSERS[dataset]
    frames = [parse(p) for p in paths]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    tcol = _TIME_COL.get(dataset, "open_time")
    n_before = len(df)
    df = df.sort_values(tcol).drop_duplicates(tcol, keep="last").reset_index(drop=True)
    out = parquet_path(market, dataset, symbol, interval)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    manifest = {
        "market": market,
        "dataset": dataset,
        "symbol": symbol,
        "interval": interval,
        "rows": len(df),
        "duplicates_dropped": n_before - len(df),
        "first": str(df[tcol].iloc[0]),
        "last": str(df[tcol].iloc[-1]),
        "sources": {k: _sha256(p) for k, p in zip(keys, paths)},
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def data_fingerprint(paths: list[Path]) -> str:
    """Hash of the source checksums behind the given Parquet files: identifies the exact data used."""
    h = hashlib.sha256()
    for p in sorted(paths):
        m = json.loads(Path(p).with_suffix(".manifest.json").read_text())
        h.update(json.dumps(m["sources"], sort_keys=True).encode())
    return h.hexdigest()[:16]


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _load(path: Path, tcol: str, start, end) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not built yet; run `cqs download`")
    end_ts = guard.resolve_end(end)
    start_ts = _utc(start) if start is not None else None
    q = f"SELECT * FROM read_parquet('{path.as_posix()}') WHERE {tcol} <= ?"
    params: list = [end_ts.to_pydatetime()]
    if start_ts is not None:
        q += f" AND {tcol} >= ?"
        params.append(start_ts.to_pydatetime())
    q += f" ORDER BY {tcol}"
    df = duckdb.execute(q, params).df()
    if not df.empty:
        df[tcol] = pd.to_datetime(df[tcol], utc=True)
    return guard.enforce(df, tcol)


def load_klines(symbol: str, interval: str = "1h", start=None, end=None, market: str = "futures/um",
                strip_zombies: bool = True, dataset: str = KLINES) -> pd.DataFrame:
    """Bars for one symbol, holdout-guarded, with the post-delisting zombie tail removed.

    Note: the zombie tail is judged on the full stored history, so a contract that was delisted after
    `end` is never cut early.
    """
    df = _load(parquet_path(market, dataset, symbol, interval), "open_time", start, end)
    if strip_zombies and not df.empty:
        full_tail_end = _last_traded_open_time(market, dataset, symbol, interval)
        if full_tail_end is not None:
            df = df[df["open_time"] <= full_tail_end]
    return df.reset_index(drop=True)


def _last_traded_open_time(market, dataset, symbol, interval):
    path = parquet_path(market, dataset, symbol, interval)
    full = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close", "volume"])
    traded = strip_zombie_tail(full)
    return None if traded.empty else pd.Timestamp(traded["open_time"].iloc[-1])


def load_funding(symbol: str, start=None, end=None, market: str = "futures/um") -> pd.DataFrame:
    return _load(parquet_path(market, FUNDING, symbol), "funding_time", start, end).reset_index(drop=True)


def load_metrics(symbol: str, start=None, end=None, market: str = "futures/um") -> pd.DataFrame:
    """OI and positioning snapshots. Use `available_time`, not `create_time`, when joining to decisions."""
    return _load(parquet_path(market, METRICS, symbol), "create_time", start, end).reset_index(drop=True)
