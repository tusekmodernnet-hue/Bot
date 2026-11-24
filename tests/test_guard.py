import pandas as pd
import pytest

from cqs import config
from cqs.data import guard, store
from cqs.research import registry


def test_explicit_holdout_end_is_refused():
    with pytest.raises(guard.HoldoutAccessError):
        guard.resolve_end("2025-06-01")


def test_default_end_stops_before_holdout():
    assert guard.resolve_end(None) < config.HOLDOUT_START


def test_final_test_needs_env_and_reason(monkeypatch):
    with pytest.raises(guard.HoldoutAccessError):
        with guard.final_test("final test of candidate X"):
            pass
    monkeypatch.setenv("CQS_ALLOW_HOLDOUT", "1")
    with pytest.raises(ValueError):
        with guard.final_test("x"):
            pass
    with guard.final_test("final test of candidate X"):
        assert guard.resolve_end("2026-01-01") == pd.Timestamp("2026-01-01", tz="UTC")
    events = [r["event"] for r in registry.read() if r["type"] == "event"]
    assert events == ["holdout_opened", "holdout_closed"]
    assert not guard.holdout_open()


def test_loader_never_returns_holdout_rows():
    idx = pd.date_range("2025-05-31 20:00", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"open_time": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    path = store.parquet_path("futures/um", store.KLINES, "TESTUSDT", "1h")
    path.parent.mkdir(parents=True)
    df.to_parquet(path, index=False)
    out = store.load_klines("TESTUSDT", "1h")
    assert len(out) == 4 and out.open_time.max() < config.HOLDOUT_START
