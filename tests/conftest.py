import numpy as np
import pandas as pd
import pytest

from cqs import config


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Every test gets its own data dir and registry; the real ones are never touched."""
    monkeypatch.setattr(config, "PARQUET_DIR", tmp_path / "parquet")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.delenv("CQS_ALLOW_HOLDOUT", raising=False)
    yield


def make_bars(closes, start="2021-01-01", freq="1h", opens=None, highs=None, lows=None, asset="A"):
    """Single-asset OHLC frames. By default open = previous close (no gaps), high/low = max/min(open, close)."""
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else np.r_[c[0], c[:-1]]
    h = np.asarray(highs, dtype=float) if highs is not None else np.maximum(o, c)
    l = np.asarray(lows, dtype=float) if lows is not None else np.minimum(o, c)
    idx = pd.date_range(start, periods=len(c), freq=freq, tz="UTC")
    f = lambda x: pd.DataFrame({asset: x}, index=idx)
    return f(o), f(h), f(l), f(c)
