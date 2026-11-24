"""Benchmarks every candidate strategy must beat (Stage 1 protocol), with fixed, untuned definitions:

- BENCH/buy_and_hold: long 1x the perpetual, rebalanced to 1x equity at every daily close, funding paid.
  (Never rebalancing is not a usable benchmark on perpetuals: funding paid while the notional is large is not
  recovered when the price falls, so leverage drifts up. Measured on BTCUSDT: a 1x long opened on
  2020-01-01 and never rebalanced was wiped out by funding and the drawdown on 2022-06-18.)
- BENCH/trend_sma200: long 1x when the daily close is above its 200-day simple moving average, else flat.
  Decided at the daily close (00:00 UTC), executed at the next hourly open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.backtest import engine, metrics
from cqs.backtest.costs import CostModel
from cqs.data import store
from cqs.research import registry, stats


def load_panel(symbols: list[str], interval: str = "1h", start=None, end=None):
    """Wide OHLC frames (time x symbol) and funding (settlement time x symbol), holdout-guarded."""
    o, h, l, c, f = {}, {}, {}, {}, {}
    for s in symbols:
        k = store.load_klines(s, interval, start=start, end=end).set_index("open_time")
        o[s], h[s], l[s], c[s] = k["open"], k["high"], k["low"], k["close"]
        fr = store.load_funding(s, start=start, end=end)
        f[s] = fr.set_index("funding_time")["rate"]
    frames = [pd.DataFrame(x).sort_index() for x in (o, h, l, c)]
    funding = pd.DataFrame(f).sort_index()
    return (*frames, funding)


def _last_bar_of_day(px: pd.Series) -> pd.Series:
    """Hourly closes of the last bar of each UTC day (that bar's close is the daily close)."""
    px = px.dropna()
    return px.groupby(px.index.normalize()).tail(1)


def buy_and_hold_weights(close: pd.DataFrame) -> pd.DataFrame:
    """1/N long per asset, reset to target at every daily close."""
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for s in close.columns:
        w.loc[_last_bar_of_day(close[s]).index, s] = 1.0 / close.shape[1]
    return w


def trend_sma_weights(close: pd.DataFrame, days: int = 200) -> pd.DataFrame:
    """1/N long per asset above its SMA, decided on the last hourly bar of each UTC day."""
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for s in close.columns:
        daily = _last_bar_of_day(close[s])
        sma = daily.rolling(days, min_periods=days).mean()
        sig = (daily > sma).astype(float).where(sma.notna())
        w.loc[sig.dropna().index, s] = sig.dropna().to_numpy() / close.shape[1]
    return w


def run_benchmarks(symbols: list[str], start=None, end=None, costs: CostModel = CostModel(), log: bool = True):
    o, h, l, c, funding = load_panel(symbols, "1h", start, end)
    fingerprint = store.data_fingerprint(
        [store.parquet_path("futures/um", store.KLINES, s, "1h") for s in symbols]
        + [store.parquet_path("futures/um", store.FUNDING, s) for s in symbols]
    )
    out = {}
    for name, w in (("buy_and_hold", buy_and_hold_weights(c)), ("trend_sma200", trend_sma_weights(c))):
        res = engine.run(o, h, l, c, w, costs, funding=funding)
        m = metrics.compute(res)
        r = metrics.daily_returns(res.equity)
        m["psr_vs_zero"] = stats.probabilistic_sharpe(r.to_numpy())
        if log:
            registry.log_experiment("BENCH", name, {"symbols": symbols, "costs": costs.__dict__ | {"slippage_bps": dict(costs.slippage_bps)}},
                                    m, fingerprint, (str(c.index[0]), str(c.index[-1])),
                                    sharpe_per_period=stats.sharpe_per_period(r.to_numpy()))
        out[name] = (res, m)
    return out
