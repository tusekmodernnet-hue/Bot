"""Round 13 building blocks on cases with known answers and the runner end to end on a synthetic Hyperliquid store
(R13A: R9A's rule on new Hyperliquid perpetuals)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cqs import config
from cqs.data import hyperliquid as HL
from cqs.data import store
from cqs.strategies import hl_listings, listing_risk

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


def test_binance_names_with_the_k_prefix():
    b = {"1000PEPEUSDT", "SHIBUSDT", "WIFUSDT"}
    assert hl_listings.binance_symbol("kPEPE", b) == "1000PEPEUSDT"
    assert hl_listings.binance_symbol("kSHIB", b) == "SHIBUSDT"              # no 1000x contract: plain name
    assert hl_listings.binance_symbol("WIF", b) == "WIFUSDT"
    assert hl_listings.binance_symbol("HYPE", b) is None
    assert hl_listings.binance_symbol("k", b) is None


def test_events_drop_btc_old_and_renamed_tokens():
    first_hl = {"BTC": ts("2023-03-01"), "OLD": ts("2024-03-01"), "SAME": ts("2024-03-01"), "HLFIRST": ts("2024-03-01"),
                "RENDER": ts("2024-03-01"), "LATE": ts("2025-05-30"), "NEW": ts("2024-04-01")}
    bfirst = {"OLDUSDT": ts("2024-01-30"),              # 31 days before D0: the perpetual market existed
              "SAMEUSDT": ts("2024-01-31"),             # 30 days before: still new
              "HLFIRSTUSDT": ts("2024-06-01"),          # listed later on Binance
              "RENDERUSDT": ts("2024-07-01")}           # a rename of an old token
    ev, why = hl_listings.events(first_hl, bfirst, ts("2025-05-29"))
    assert sorted(ev.index) == ["HLFIRST", "NEW", "SAME"]
    assert why["btc"] == ["BTC"] and why["old_token"] == ["OLD"] and why["renamed_token"] == ["RENDER"]
    assert why["after_window"] == ["LATE"] and ev["NEW"] == ts("2024-04-01")


def test_near_misses_flag_other_unit_multipliers():
    assert hl_listings.near_misses(["kMOG", "WIF", "HYPE"], {"1000000MOGUSDT", "WIFUSDT"}) == {"kMOG": ["1000000MOGUSDT"]}


def test_first_traded_day_needs_volume():
    k = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=4, freq="1D", tz=UTC), "volume": [0, 0, 5, 7]})
    assert hl_listings.first_traded_days({"X": k, "Z": k.assign(volume=0)}) == {"X": ts("2024-01-03")}


def test_basket_is_point_in_time_aged_liquid_top20_without_btc():
    idx = pd.date_range("2024-01-01", periods=130, freq="1D", tz=UTC)
    cols = ["BTC"] + [f"A{i}" for i in range(22)] + ["THIN", "YOUNG"]
    close = pd.DataFrame(1.0, index=idx, columns=cols)
    close.loc[idx[:60], "YOUNG"] = np.nan
    qv = pd.DataFrame({c: 3e6 + i * 1e5 for i, c in enumerate(cols)}, index=idx)
    qv["THIN"] = 1e6
    b = hl_listings.basket(close, qv, qv)
    assert "BTC" not in b.columns and "THIN" not in b.columns
    assert not b.iloc[:89].any().any() and b.iloc[89:].sum(axis=1).eq(20).all()   # 90 days of history first
    assert "YOUNG" not in b.columns                                                 # 70 days old at the end
    assert "A0" not in b.columns and "A1" not in b.columns                          # the two smallest of 22 alts


def test_r9a_default_volume_filter_is_unchanged_and_2m_admits_smaller_listings():
    idx = pd.date_range("2024-01-01", periods=50, freq="1D", tz=UTC)
    close = pd.DataFrame({"N": np.r_[np.full(5, np.nan), np.full(45, 4.0)], "A": 1.0}, index=idx)
    opn = close.shift(1).fillna(close).where(close.notna())
    qv = (close * 0 + 5e6)
    ev = pd.Series({"N": idx[5]}, dtype=object)
    basket = pd.DataFrame({"A": True}, index=idx)
    w0, h0, _ = listing_risk.r9a(opn, close, qv, ev, basket, 1.0)
    w1, h1, _ = listing_risk.r9a(opn, close, qv, ev, basket, 1.0, min_qv=20e6)
    pd.testing.assert_frame_equal(w0, w1)
    assert not h0["N"].any()                                     # 5M < 20M
    _, h2, _ = listing_risk.r9a(opn, close, qv, ev, basket, 1.0, min_qv=hl_listings.MIN_QUOTE_VOLUME)
    assert h2["N"].iloc[6:36].all() and not h2["N"].iloc[36:].any()


def test_funding_is_put_on_one_hourly_index():
    from cqs.research import run_round13 as R13
    t = pd.to_datetime([ts("2024-01-01 00:00:00.076"), ts("2024-01-01 00:59:59.990"), ts("2024-01-01 02:00:00.020")])
    fr = pd.DataFrame({"funding_time": t, "rate": [1e-5, 2e-5, 3e-5]})
    s = R13.funding_hourly(fr)
    assert list(s.index) == [ts("2024-01-01 00:00:01"), ts("2024-01-01 01:00:01"), ts("2024-01-01 02:00:01")]
    assert list(s) == pytest.approx([1e-5, 2e-5, 3e-5])


def test_verdict_rule():
    from cqs.research import run_round13 as R13
    assert R13.verdict(1.2, 0.95, 0.8, 40, 0.0) == "replicated"
    assert R13.verdict(1.2, 0.85, 0.8, 40, 0.0) == "inconclusive"
    assert R13.verdict(1.2, 0.95, -0.1, 40, 0.0) == "inconclusive"
    assert R13.verdict(-0.1, 0.30, -0.5, 40, 0.0) == "not_replicated"
    assert R13.verdict(1.2, 0.95, 0.8, 29, 0.0) == "too_few_shorts"
    assert R13.verdict(1.2, 0.95, 0.8, 40, 0.11) == "survivorship_risk"


# ------------------------------------------------------------------ runner end to end
def _write_store(root, n_new=36, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", "2025-05-31", freq="1D", tz=UTC)
    meta, binance = [], []
    coins = {"BTC": (0, 0.0, 6e8)} | {f"A{i}": (0, 0.0, 3e7) for i in range(6)} | \
        {f"N{j}": (150 + 20 * j, -0.004, 6e6) for j in range(n_new)} | {"OLDT": (200, -0.004, 6e6)}
    binance.append(("OLDTUSDT", "2021-01-01"))
    for a, (coin, (start, drift, qv)) in enumerate(coins.items()):
        meta.append({"asset": a, "name": coin, "is_delisted": False})
        live = idx[start:]
        c = 10 * np.exp(np.cumsum(rng.normal(drift, 0.05, len(live))))
        o = np.r_[c[0], c[:-1]]
        k = pd.DataFrame({"open_time": live, "open": o, "high": np.maximum(o, c) * 1.01,
                          "low": np.minimum(o, c) * 0.99, "close": c, "volume": qv / c, "quote_volume": qv,
                          "trades": 100})
        p = store.parquet_path(HL.MARKET, store.KLINES, coin, "1d")
        p.parent.mkdir(parents=True, exist_ok=True)
        k.to_parquet(p, index=False)
        p.with_suffix(".manifest.json").write_text(json.dumps({"sources": {coin: "k"}}))
        ft = pd.date_range(live[0], idx[-1] + pd.Timedelta(hours=23), freq="1h", tz=UTC) + pd.Timedelta(milliseconds=70)
        f = pd.DataFrame({"funding_time": ft, "rate": rng.normal(1e-5, 2e-5, len(ft)), "premium": 0.0})
        q = store.parquet_path(HL.MARKET, store.FUNDING, coin)
        q.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(q, index=False)
        q.with_suffix(".manifest.json").write_text(json.dumps({"sources": {coin: "f"}}))
    HL.meta_path().write_text(json.dumps({"universe": meta}))
    HL.summary_path().write_text(json.dumps({"coins_in_meta": len(meta), "missing": [], "missing_share": 0.0}))
    return binance


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    from cqs.research import run_round13 as R13
    monkeypatch.setattr(config, "PARQUET_DIR", tmp_path / "parquet")
    monkeypatch.setattr(config, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(R13, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(R13, "preregistered", lambda: True)
    monkeypatch.setattr(R13.R7, "rf_daily", lambda index: pd.Series(0.02, index=index))
    binance = _write_store(tmp_path)
    bf = tmp_path / "binance_first.csv"
    pd.DataFrame(binance, columns=["symbol", "first_traded_day"]).to_csv(bf, index=False)
    monkeypatch.setattr(R13, "BINANCE_FIRST", bf)
    return R13, tmp_path


def test_round13_runs_end_to_end_logs_one_trial_and_never_reruns(synthetic):
    from cqs.research import registry
    R13, tmp = synthetic
    assert R13.main([]) == 0
    out = json.loads((tmp / "results" / "R13A.json").read_text())
    # 36 new coins plus the 6 alts listed on the first day (no Binance perpetual either): no basket exists for
    # those yet, so they cannot be shorted
    assert out["events"] == 42 and out["excluded"]["old_token"] == 1 and out["excluded"]["btc"] == 1
    assert out["shorts_traded"] > 0 and out["verdict"] in {"replicated", "inconclusive", "not_replicated",
                                                           "too_few_shorts"}
    assert out["excess"]["start"] >= "2023-05-31"           # nothing before the basket has 90-day-old members
    assert (tmp / "results" / "R13A_daily.csv").exists()
    summ = json.loads((tmp / "results" / "R13_summary.json").read_text())
    assert summ["candidate"] is False and summ["n_trials_program"] == 1
    ex = [r for r in registry.read() if r.get("type") == "experiment"]
    assert [r["family"] for r in ex] == ["R13A"]
    assert R13.main([]) == 0
    assert len([r for r in registry.read() if r.get("type") == "experiment"]) == 1


def test_round13_refuses_traded_contracts_without_funding(synthetic):
    R13, tmp = synthetic
    store.parquet_path(HL.MARKET, store.FUNDING, "N3").unlink()
    with pytest.raises(SystemExit, match="no funding data"):
        R13.main([])
