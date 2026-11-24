"""Position sizing and risk caps (Stage 1 defaults). All inputs are known at the decision time."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAJORS = ("BTCUSDT", "ETHUSDT")


def inverse_vol(signal: pd.DataFrame, returns: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """Scale each non-zero signal by 1 / trailing daily volatility (equal risk per position)."""
    vol = returns.rolling(lookback, min_periods=lookback).std()
    w = signal / vol
    return w.where(signal != 0, 0.0).where(signal.notna())


def normalize_gross(w: pd.DataFrame) -> pd.DataFrame:
    """Rescale each row to gross exposure 1 (rows with no position stay 0)."""
    g = w.abs().sum(axis=1)
    return w.div(g.where(g > 0), axis=0).fillna(0.0).where(w.notna())


def vol_target(w: pd.DataFrame, returns: pd.DataFrame, target_annual: float = 0.15, lookback: int = 60,
               max_gross: float = 2.0) -> pd.DataFrame:
    """Scale a weight path so its trailing realised volatility hits the target.

    The scale at day t uses only portfolio returns up to t: weights held on day s-1 times asset returns on s.
    """
    held = w.fillna(0.0).shift(1)
    port = (held * returns.fillna(0.0)).sum(axis=1)
    vol = port.rolling(lookback, min_periods=lookback).std() * np.sqrt(365)
    scale = (target_annual / vol).replace([np.inf, -np.inf], np.nan)
    out = w.mul(scale, axis=0)
    gross = out.abs().sum(axis=1)
    over = (gross / max_gross).clip(lower=1.0)
    return out.div(over, axis=0)


def cap_positions(w: pd.DataFrame, major_cap: float = 1.0, alt_cap: float = 0.25) -> pd.DataFrame:
    caps = pd.Series({c: (major_cap if c in MAJORS else alt_cap) for c in w.columns})
    return w.clip(lower=-caps, upper=caps, axis=1)


def beta_to(returns: pd.DataFrame, ref: str = "BTCUSDT", lookback: int = 60) -> pd.DataFrame:
    r_ref = returns[ref]
    cov = returns.rolling(lookback, min_periods=lookback).cov(r_ref)
    var = r_ref.rolling(lookback, min_periods=lookback).var()
    return cov.div(var, axis=0)
