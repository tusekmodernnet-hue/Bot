import numpy as np
import pandas as pd

from cqs.data.universe import UniverseRules, eligible


def panel(days=200):
    idx = pd.date_range("2021-01-01", periods=days, freq="1D", tz="UTC")
    close = pd.DataFrame(index=idx, data={"OLDUSDT": 1.0, "NEWUSDT": np.nan, "DEADUSDT": 1.0, "ILLIQUSDT": 1.0})
    close.loc[idx[100]:, "NEWUSDT"] = 1.0  # listed on day 100
    close.loc[idx[150]:, "DEADUSDT"] = np.nan  # delisted on day 150
    qv = close.notna() * 50e6
    qv["ILLIQUSDT"] = 1e6
    vol = close.notna() * 1.0
    return {"close": close, "quote_volume": qv.where(close.notna()), "volume": vol.where(close.notna())}


def test_rules_age_delisting_and_liquidity():
    p = panel()
    e = eligible(p, UniverseRules(min_age_days=90, liquidity_window_days=30, min_avg_quote_volume=20e6, top_n=None))
    idx = p["close"].index
    assert e.loc[idx[120], "OLDUSDT"] and e.loc[idx[120], "DEADUSDT"]
    assert not e.loc[idx[120], "NEWUSDT"]  # only 21 days old
    assert e.loc[idx[195], "NEWUSDT"]  # 96 days old
    assert not e.loc[idx[160], "DEADUSDT"]  # delisted
    assert not e["ILLIQUSDT"].any()


def test_no_future_information():
    p = panel(200)
    short = {k: v.iloc[:130] for k, v in p.items()}
    e_full = eligible(p)
    e_short = eligible(short)
    pd.testing.assert_frame_equal(e_full.iloc[:130], e_short)
