"""Benchmarks measured over a given window (a family whose out-of-sample period differs from 2022-01..2025-05)."""

from __future__ import annotations

import pandas as pd

from cqs.backtest import metrics
from cqs.backtest.costs import CostModel
from cqs.research import benchmarks, walkforward


def benchmarks_in_window(start: str, end: str, costs: CostModel) -> dict:
    """Buy & hold and SMA200 on BTC and on BTC+ETH, simulated on the full development data (so the SMA has its
    history) and measured on [start, end]. Not logged: these are the already-registered benchmarks, sliced."""
    out = {}
    for syms in (["BTCUSDT"], ["BTCUSDT", "ETHUSDT"]):
        for name, (res, _m) in benchmarks.run_benchmarks(syms, costs=costs, log=False).items():
            r = metrics.daily_returns(res.equity)
            r = r[(r.index >= pd.Timestamp(start, tz="UTC")) & (r.index <= pd.Timestamp(end, tz="UTC"))]
            out[f"{'+'.join(syms)}/{name}"] = walkforward.summary(r)
    return out
