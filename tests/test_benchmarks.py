import numpy as np
import pandas as pd

from cqs.research.benchmarks import buy_and_hold_weights, trend_sma_weights


def hourly(days=260, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=days * 24, freq="1h", tz="UTC")
    return pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx))))}, index=idx)


def test_decisions_only_on_last_bar_of_day():
    c = hourly()
    for w in (buy_and_hold_weights(c), trend_sma_weights(c, days=50)):
        decided = w["A"].dropna().index
        assert (decided.hour == 23).all()


def test_trend_weights_ignore_future_prices():
    c = hourly()
    w1 = trend_sma_weights(c, days=50)
    c2 = c.copy()
    c2.iloc[-24 * 30:] *= 3.0  # change the last 30 days only
    w2 = trend_sma_weights(c2, days=50)
    cut = c.index[-24 * 30] - pd.Timedelta(hours=1)
    pd.testing.assert_frame_equal(w1.loc[:cut], w2.loc[:cut])
