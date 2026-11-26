"""Perpetual -> Binance spot pair mapping, spot listing dates and the round-7 data download.

A USDⓈ-M perpetual is matched to the spot pair with the same symbol, or to the pair without the contract-size
prefix (1000PEPEUSDT -> PEPEUSDT with a multiplier of 1000). The match is only a name match: research code must
also check that the two prices agree (`price_consistent`) before treating the pair as the same asset.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys

import pandas as pd

from cqs import config
from cqs.data import archive, store

PREFIXES = (("1000000", 1e6), ("100000", 1e5), ("10000", 1e4), ("1000", 1e3), ("1M", 1e6))
SPOT_LIST = config.DATA_DIR / "spot_symbols.txt"


def candidates(perp: str) -> list[tuple[str, float]]:
    """Spot names to try for a perpetual, in order, with the contract multiplier (perp price / spot price)."""
    out = [(perp, 1.0)]
    for p, m in PREFIXES:
        if perp.startswith(p) and len(perp) > len(p) + 4:
            out.append((perp[len(p):], m))
            break
    return out


def spot_symbols(refresh: bool = False) -> set[str]:
    """Spot symbols with monthly klines in the archive (cached in data/spot_symbols.txt)."""
    if refresh or not SPOT_LIST.exists():
        syms = archive.list_symbols(market="spot", dataset="klines", period="monthly")
        SPOT_LIST.write_text("\n".join(syms) + "\n")
    return set(SPOT_LIST.read_text().split())


def match(perp: str, spot: set[str]) -> tuple[str, float] | None:
    for s, m in candidates(perp):
        if s in spot:
            return s, m
    return None


def price_consistent(perp_close: pd.Series, spot_close: pd.Series, mult: float, window: int = 7,
                     tol: float = 0.10) -> pd.Series:
    """True on day t when the median of perp / (mult * spot) over the last `window` days with both prices is
    within `tol` of 1. Uses closes up to t only. A wrong name match (another asset) fails it."""
    ratio = perp_close / (mult * spot_close.reindex(perp_close.index))
    med = ratio.rolling(window, min_periods=window).median()
    return (med - 1.0).abs() <= tol


def first_traded_day(symbol: str, market: str) -> pd.Timestamp | None:
    """First day with a daily bar with volume (holdout-guarded load)."""
    try:
        k = store.load_klines(symbol, "1d", market=market)
    except FileNotFoundError:
        return None
    k = k[k["volume"] > 0]
    return None if k.empty else pd.Timestamp(k["open_time"].iloc[0]).normalize()


def _build(args) -> tuple[str, str]:
    market, dataset, sym, interval = args
    path = store.parquet_path(market, dataset, sym, interval)
    if path.exists():
        return sym, "kept"
    try:
        store.build(market, dataset, sym, interval, workers=8)
        return sym, "built"
    except FileNotFoundError as e:
        return sym, f"missing: {e}"


def round7_perps() -> tuple[list[str], list[str]]:
    """Perpetuals round 7 needs: the ever-top-50 set (R7A, R7B) and every perpetual whose first traded day is
    inside the research period (R7C candidates; liquidity is judged later, point-in-time)."""
    top50 = (config.DATA_DIR / "universe_ever_top50.txt").read_text().split()
    listed = []
    for s in sorted(p.name for p in (config.PARQUET_DIR / "futures_um" / store.KLINES).iterdir()):
        d = first_traded_day(s, "futures/um")
        if d is not None and d > pd.Timestamp("2020-01-01", tz="UTC"):
            listed.append(s)
    return top50, listed


def download_round7(workers: int = 6) -> dict:
    top50, listed = round7_perps()
    spot = spot_symbols(refresh=True)
    jobs = []
    for p in sorted(set(top50) | set(listed)):
        m = match(p, spot)
        if m is not None:
            jobs.append(("spot", store.KLINES, m[0], "1d"))
    for p in listed:
        jobs.append(("futures/um", store.FUNDING, p, None))
    done = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, (sym, status) in enumerate(ex.map(_build, jobs)):
            done[f"{jobs[i][0]}:{jobs[i][1]}:{sym}"] = status
            if i % 25 == 0:
                print(i, len(jobs), sym, status, flush=True)
    return done


if __name__ == "__main__":
    res = download_round7()
    bad = {k: v for k, v in res.items() if v.startswith("missing")}
    print("jobs", len(res), "built", sum(v == "built" for v in res.values()), "missing", len(bad))
    for k, v in list(bad.items())[:50]:
        print(k, v)
    sys.exit(0)
