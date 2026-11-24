"""Performance metrics. Return-based metrics are computed on daily returns (365 trading days a year)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.backtest.engine import BacktestResult

DAYS_PER_YEAR = 365


def daily_returns(equity: pd.Series, initial_equity: float = 1.0) -> pd.Series:
    """Daily returns from bar equity (equity at each day's last bar close)."""
    eod = equity.resample("1D").last().dropna()
    prev = eod.shift(1)
    prev.iloc[0] = initial_equity
    return eod / prev - 1.0


def sharpe(r: pd.Series, periods: int = DAYS_PER_YEAR) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(r: pd.Series, periods: int = DAYS_PER_YEAR) -> float:
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else float("nan")


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def drawdown_durations(equity: pd.Series) -> list[pd.Timedelta]:
    """Length of each underwater spell, from the peak to the recovery (or the end of data)."""
    peak = equity.cummax()
    under = equity < peak
    out: list[pd.Timedelta] = []
    start = None
    prev_t = None
    for t, u in under.items():
        if u and start is None:
            start = prev_t if prev_t is not None else t  # the peak is the last bar before going under
        elif not u and start is not None:
            out.append(t - start)
            start = None
        prev_t = t
    if start is not None:
        out.append(equity.index[-1] - start)
    return out


def cagr(equity: pd.Series, initial_equity: float = 1.0) -> float:
    years = (equity.index[-1] - equity.index[0]) / pd.Timedelta(days=365.25)
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / initial_equity) ** (1.0 / years) - 1.0)


def compute(result: BacktestResult) -> dict:
    eq = result.equity
    r = daily_returns(eq, result.initial_equity)
    dd = drawdown(pd.concat([pd.Series([result.initial_equity], index=[eq.index[0] - result.bar]), eq]))
    durations = drawdown_durations(eq)
    t = result.trades
    wins = t.loc[t.pnl > 0, "pnl"]
    losses = t.loc[t.pnl <= 0, "pnl"]
    years = max((eq.index[-1] - eq.index[0]) / pd.Timedelta(days=365.25), 1e-9)
    gross = result.positions.abs().sum(axis=1)
    c = cagr(eq, result.initial_equity)
    mdd = float(dd.min())
    var5 = float(r.quantile(0.05)) if len(r) else float("nan")
    return {
        "start": str(eq.index[0]),
        "end": str(eq.index[-1]),
        "total_return": float(eq.iloc[-1] / result.initial_equity - 1.0),
        "cagr": c,
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": mdd,
        "calmar": float(c / abs(mdd)) if mdd < 0 else float("nan"),
        "avg_drawdown_days": float(np.mean([d / pd.Timedelta(days=1) for d in durations])) if durations else 0.0,
        "max_drawdown_days": float(max(d / pd.Timedelta(days=1) for d in durations)) if durations else 0.0,
        "n_trades": int(len(t)),
        "trades_per_year": float(len(t) / years),
        "win_rate": float(len(wins) / len(t)) if len(t) else float("nan"),
        "avg_win_roe": float(t.loc[t.pnl > 0, "return_on_equity"].mean()) if len(wins) else float("nan"),
        "avg_loss_roe": float(t.loc[t.pnl <= 0, "return_on_equity"].mean()) if len(losses) else float("nan"),
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("nan"),
        "expectancy_roe": float(t["return_on_equity"].mean()) if len(t) else float("nan"),
        "exposure": float((gross > 0).mean()),
        "avg_gross_leverage": float((gross / eq).mean()),
        "turnover_per_year": float(result.traded_notional / result.initial_equity / years),
        "costs_paid": float(result.costs_paid),
        "funding_paid": float(result.funding_paid),
        "daily_var_5": var5,
        "daily_cvar_5": float(r[r <= var5].mean()) if len(r) else float("nan"),
        "worst_day": float(r.min()) if len(r) else float("nan"),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "n_days": int(len(r)),
    }
