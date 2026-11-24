"""Data-quality checks for bar data.

The checks that matter most in the Binance archive:
- zombie bars: after a delisting the archive keeps emitting bars with a flat price and zero volume.
  Treating them as real history invents a period of zero volatility.
- gaps and duplicates in the time index.
- OHLC bars that are internally inconsistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from cqs import config


@dataclass
class QualityReport:
    symbol: str
    interval: str
    n_bars: int
    first_bar: str | None
    last_bar: str | None
    last_traded_bar: str | None
    zombie_tail_bars: int
    missing_bars: int
    missing_blocks: list[tuple[str, int]] = field(default_factory=list)
    duplicate_bars: int = 0
    ohlc_inconsistent: int = 0
    nonpositive_prices: int = 0
    zero_volume_bars: int = 0
    max_abs_log_return: float = float("nan")

    @property
    def is_delisted(self) -> bool:
        return self.zombie_tail_bars > 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["is_delisted"] = self.is_delisted
        return d


def zombie_tail_length(bars: pd.DataFrame) -> int:
    """Number of trailing bars with zero volume and a flat price (O=H=L=C, unchanged).

    These are the bars the archive emits after a contract stops trading.
    """
    if bars.empty:
        return 0
    vol = bars["volume"].to_numpy()
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    flat = (vol == 0) & (o == h) & (h == l) & (l == c)
    if not flat[-1]:
        return 0
    # index of the last bar that is not flat
    not_flat = np.flatnonzero(~flat)
    return len(bars) if not_flat.size == 0 else len(bars) - 1 - int(not_flat[-1])


def strip_zombie_tail(bars: pd.DataFrame, min_tail: int = 1) -> pd.DataFrame:
    """Drop the zombie tail. `min_tail` guards against cutting a single quiet bar of a live contract."""
    n = zombie_tail_length(bars)
    if n >= min_tail and n > 0:
        return bars.iloc[: len(bars) - n].copy()
    return bars


def check_bars(bars: pd.DataFrame, symbol: str, interval: str) -> QualityReport:
    """Quality report for a bar frame with columns open_time, open, high, low, close, volume."""
    if bars.empty:
        return QualityReport(symbol, interval, 0, None, None, None, 0, 0)
    step = config.INTERVAL_TO_TIMEDELTA[interval]
    b = bars.sort_values("open_time")
    t = b["open_time"]
    dups = int(t.duplicated().sum())
    b = b.drop_duplicates("open_time")
    full = pd.date_range(b["open_time"].iloc[0], b["open_time"].iloc[-1], freq=step)
    missing = full.difference(pd.DatetimeIndex(b["open_time"]))
    blocks: list[tuple[str, int]] = []
    if len(missing):
        s = pd.Series(missing)
        grp = (s.diff() != step).cumsum()
        blocks = [(str(g.iloc[0]), int(len(g))) for _, g in s.groupby(grp)]
    hi_bad = b["high"] < b[["open", "close"]].max(axis=1)
    lo_bad = b["low"] > b[["open", "close"]].min(axis=1)
    nonpos = (b[["open", "high", "low", "close"]] <= 0).any(axis=1)
    tail = zombie_tail_length(b)
    traded = b.iloc[: len(b) - tail] if tail else b
    lr = np.diff(np.log(traded["close"].to_numpy())) if len(traded) > 1 else np.array([np.nan])
    return QualityReport(
        symbol=symbol,
        interval=interval,
        n_bars=len(b),
        first_bar=str(b["open_time"].iloc[0]),
        last_bar=str(b["open_time"].iloc[-1]),
        last_traded_bar=str(traded["open_time"].iloc[-1]) if len(traded) else None,
        zombie_tail_bars=tail,
        missing_bars=len(missing),
        missing_blocks=blocks[:20],
        duplicate_bars=dups,
        ohlc_inconsistent=int((hi_bad | lo_bad).sum()),
        nonpositive_prices=int(nonpos.sum()),
        zero_volume_bars=int((b["volume"] == 0).sum()),
        max_abs_log_return=float(np.nanmax(np.abs(lr))) if lr.size else float("nan"),
    )
