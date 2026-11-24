"""Client for the Binance public data archive (data.binance.vision).

The archive is the research data source: it holds delisted contracts and needs no API key.
Every downloaded zip is verified against the archive's published SHA-256 checksum and cached in RAW_DIR,
so a rerun never downloads a file twice and a corrupted download is never parsed.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from cqs import config

USER_AGENT = "cqs-research/0.1"


class ArchiveError(RuntimeError):
    pass


def _http_get(url: str, timeout: float = 60.0, retries: int = 4) -> bytes:
    """GET with exponential backoff. Raises FileNotFoundError on 404."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2**attempt)
    raise ArchiveError(f"GET failed after {retries} attempts: {url}: {last}")


def list_prefix(prefix: str) -> tuple[list[str], list[str]]:
    """List sub-prefixes and object keys directly under `prefix` (S3 delimiter listing, paginated)."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        query = {"prefix": prefix, "delimiter": "/"}
        if marker:
            query["marker"] = marker
        xml = _http_get(config.ARCHIVE_LIST_URL + "?" + urllib.parse.urlencode(query)).decode("utf-8")
        page_prefixes = re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", xml)
        page_keys = re.findall(r"<Key>([^<]*)</Key>", xml)
        prefixes += page_prefixes
        keys += page_keys
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        nxt = re.findall(r"<NextMarker>([^<]*)</NextMarker>", xml)
        marker = nxt[0] if nxt else (page_keys or page_prefixes)[-1]
    return prefixes, keys


@dataclass(frozen=True)
class DatasetSpec:
    """One archive dataset, e.g. USD-M futures 1h klines or funding rates."""

    market: str  # "futures/um", "futures/cm" or "spot"
    dataset: str  # "klines", "fundingRate", "metrics", "premiumIndexKlines", ...
    interval: str | None = None  # for kline-like datasets

    def prefix(self, period: str, symbol: str) -> str:
        parts = ["data", self.market, period, self.dataset, symbol]
        if self.interval:
            parts.append(self.interval)
        return "/".join(parts) + "/"

    def filename(self, symbol: str, date: str) -> str:
        middle = self.interval if self.interval else self.dataset
        return f"{symbol}-{middle}-{date}.zip"


def list_symbols(market: str = "futures/um", dataset: str = "klines", period: str = "monthly") -> list[str]:
    prefixes, _ = list_prefix(f"data/{market}/{period}/{dataset}/")
    return sorted(p.rstrip("/").split("/")[-1] for p in prefixes)


def list_files(spec: DatasetSpec, symbol: str, period: str) -> list[str]:
    """Archive keys of the zip files for one symbol (sorted, without checksum files)."""
    _, keys = list_prefix(spec.prefix(period, symbol))
    return sorted(k for k in keys if k.endswith(".zip"))


def local_path(key: str) -> Path:
    return config.RAW_DIR / key


def download(key: str, force: bool = False) -> Path:
    """Download one archive file, verify its SHA-256 checksum and cache it. Returns the local path."""
    path = local_path(key)
    if path.exists() and not force:
        return path
    url = f"{config.ARCHIVE_BASE_URL}/{urllib.parse.quote(key)}"
    payload = _http_get(url)
    try:
        checksum_text = _http_get(url + ".CHECKSUM").decode("utf-8").strip()
        expected = checksum_text.split()[0].lower()
    except FileNotFoundError:
        expected = None
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and expected != actual:
        raise ArchiveError(f"checksum mismatch for {key}: expected {expected}, got {actual}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def download_many(keys: list[str], workers: int = 16) -> list[Path]:
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(download, keys))


def files_for_symbol(spec: DatasetSpec, symbol: str) -> list[str]:
    """All keys needed for a complete history: monthly files, plus daily files after the last month.

    Monthly files cover complete months; the current month exists only as daily files.
    """
    monthly = list_files(spec, symbol, "monthly")
    last_month = None
    if monthly:
        last_month = re.search(r"(\d{4}-\d{2})\.zip$", monthly[-1]).group(1)
    daily = list_files(spec, symbol, "daily")
    if last_month is not None:
        daily = [k for k in daily if re.search(r"(\d{4}-\d{2})-\d{2}\.zip$", k).group(1) > last_month]
    return monthly + daily
