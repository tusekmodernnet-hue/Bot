"""Walk-forward selection, regime breakdown and family evaluation (research/preregistration.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs.backtest import metrics


def windows(index: pd.DatetimeIndex, train_months: int = 24, test_months: int = 3, start=None):
    """(train_start, train_end, test_start, test_end) tuples, each end exclusive."""
    t0 = pd.Timestamp(start) if start is not None else index[0]
    out = []
    test_start = t0 + pd.DateOffset(months=train_months)
    while test_start < index[-1]:
        test_end = test_start + pd.DateOffset(months=test_months)
        out.append((test_start - pd.DateOffset(months=train_months), test_start, test_start, test_end))
        test_start = test_end
    return out


def ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(365)) if len(r) > 2 and sd > 0 else float("nan")


def select_oos(daily: pd.DataFrame, train_months: int = 24, test_months: int = 3, start=None):
    """Stitch out-of-sample returns: in each window, the variant with the best train Sharpe is used for the test.

    daily: date x variant matrix of net daily returns. Returns (oos returns, choices per window).
    """
    parts, choices = [], []
    for tr0, tr1, te0, te1 in windows(daily.index, train_months, test_months, start):
        train = daily[(daily.index >= tr0) & (daily.index < tr1)]
        test = daily[(daily.index >= te0) & (daily.index < te1)]
        if test.empty:
            continue
        sharpes = train.apply(ann_sharpe)
        if sharpes.isna().all():
            continue
        best = sharpes.idxmax()
        parts.append(test[best])
        choices.append({"test_start": str(te0.date()), "test_end": str(te1.date()), "variant": best,
                        "train_sharpe": float(sharpes[best])})
    oos = pd.concat(parts) if parts else pd.Series(dtype=float)
    return oos, choices


def regimes(btc_close: pd.Series) -> pd.DataFrame:
    """Pre-registered regime labels from BTC, known at each day's close."""
    r90 = btc_close / btc_close.shift(90) - 1.0
    trend = pd.Series(np.where(r90 > 0.2, "bull", np.where(r90 < -0.2, "bear", "range")), index=btc_close.index)
    trend[r90.isna()] = np.nan
    vol = btc_close.pct_change(fill_method=None).rolling(30).std() * np.sqrt(365)
    return pd.DataFrame({"trend": trend, "high_vol": vol > 0.8})


def by_regime(r: pd.Series, reg: pd.DataFrame) -> dict:
    """Sharpe and total return of a daily return series by regime; the label of day t-1 applies to day t."""
    lab = reg.shift(1).reindex(r.index)
    out = {}
    for name in ("bull", "bear", "range"):
        x = r[lab["trend"] == name]
        out[name] = {"days": int(len(x)), "sharpe": ann_sharpe(x), "total_return": float((1 + x).prod() - 1)}
    x = r[lab["high_vol"] == True]  # noqa: E712
    out["high_vol"] = {"days": int(len(x)), "sharpe": ann_sharpe(x), "total_return": float((1 + x).prod() - 1)}
    return out


def summary(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    years = len(r) / 365
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else float("nan")
    mdd = float(min(dd.min(), 0.0))
    return {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "days": int(len(r)),
            "sharpe": ann_sharpe(r), "sortino": metrics.sortino(r), "cagr": cagr, "max_drawdown": mdd,
            "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"), "total_return": float(eq.iloc[-1] - 1)}
