"""Round 6: calendar basis of quarterly futures against perpetual funding, market neutral
(research/preregistration.md, "Runda 6"). Venue-agnostic: the runner passes hourly bars of one perpetual and its
quarterly contracts.

Conventions (all pre-registered):
- One decision per day, at the close of the 23:00 UTC hourly bar (t = 00:00 UTC); the engine executes it at the
  next bar's open.
- Front contract at t: among the quarterlies listed at least WARMUP before t, with a bar at t and at least MIN_DAYS
  days to delivery, the one that delivers first. A held contract is rolled at the first decision where it no
  longer qualifies; with no qualifying contract the coin is flat.
- Annualised basis b = (F / P - 1) * 365 / D: F, P = closes of the decision bar, D = days from t to delivery.
- Trailing funding f = sum of the perpetual's funding rates whose settlement + 5 minutes lies in (t - 30 d, t],
  times 365 / 30 (venue independent: Binance settles every 8 h, Deribit every hour).
- State per coin: +1 = short basis (long perpetual, short quarterly), -1 = long basis, 0 = flat. Each leg is LEG
  of equity. Weights are emitted only when the state or the contract changes; in between, the engine holds the
  quantities (both legs drift with their prices, so the hedge stays coin for coin).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cqs import config
from cqs.data.quarterly import Contract

DECISION_HOUR = 23
BAR = pd.Timedelta(hours=1)
MIN_DAYS = 7.0
WARMUP = pd.Timedelta(days=1)
FUNDING_DAYS = 30
LEG = 0.5
DAY = pd.Timedelta(days=1)


def decision_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the bars whose close is a decision time (the 23:00 UTC bars)."""
    return np.flatnonzero(index.hour == DECISION_HOUR)


def trailing_funding(funding: pd.Series, times: pd.DatetimeIndex, days: int = FUNDING_DAYS,
                     delay: pd.Timedelta = config.METRICS_DELAY) -> np.ndarray:
    """Annualised sum of the funding known at each time over the previous `days` days (NaN before the funding
    history covers a full window)."""
    f = funding.dropna().sort_index()
    avail = pd.DatetimeIndex(f.index) + delay
    cs = np.concatenate([[0.0], np.cumsum(f.to_numpy(dtype=float))])
    start = times - pd.Timedelta(days=days)
    hi = avail.searchsorted(times, side="right")
    lo = avail.searchsorted(start, side="right")
    out = (cs[hi] - cs[lo]) * 365.0 / days
    covered = np.asarray(avail[0] <= start) if len(avail) else np.zeros(len(times), bool)
    return np.where(covered, out, np.nan)


def features(index: pd.DatetimeIndex, perp_close: pd.Series, q_close: pd.DataFrame, contracts: list[Contract],
             funding: pd.Series, min_days: float = MIN_DAYS, warmup: pd.Timedelta = WARMUP) -> pd.DataFrame:
    """One row per decision time: front contract, days to delivery, prices, basis, trailing funding, spread.
    Uses only bars that closed by the decision time and funding known by then."""
    pos = decision_positions(index)
    times = index[pos] + BAR
    P = perp_close.reindex(index).to_numpy(dtype=float)[pos]
    syms = [c.symbol for c in contracts]
    Q = q_close.reindex(index=index, columns=syms).to_numpy(dtype=float)[pos]
    order = np.argsort([c.delivery for c in contracts], kind="stable")
    front, F, D = [None] * len(pos), np.full(len(pos), np.nan), np.full(len(pos), np.nan)
    for k, t in enumerate(times):
        for j in order:
            c = contracts[j]
            days = (c.delivery - t) / DAY
            if days >= min_days and c.listed <= t - warmup and np.isfinite(Q[k, j]):
                front[k], F[k], D[k] = c.symbol, Q[k, j], days
                break
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (F / P - 1.0) * 365.0 / D
    fbar = trailing_funding(funding, times)
    return pd.DataFrame({"pos": pos, "front": pd.Series(front, index=times, dtype=object), "days": D, "F": F, "P": P,
                         "basis": b, "funding": fbar, "spread": b - fbar}, index=times)


def fronts(feat: pd.DataFrame) -> list:
    """Front contract per decision, None where there is none (pandas may store a missing string as NaN)."""
    return [x if isinstance(x, str) else None for x in feat["front"]]


def states_r6a(feat: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Always short basis on the front contract (flat only when there is none or a price is missing)."""
    fr = fronts(feat)
    ok = np.array([f is not None for f in fr], bool) & np.isfinite(feat["basis"].to_numpy(dtype=float))
    state = np.where(ok, 1, 0)
    return state, [c if s else None for s, c in zip(state, fr)]


def states_r6b(feat: pd.DataFrame, theta: float) -> tuple[np.ndarray, list]:
    """Spread s = b - f. Enter in the direction that closes it when |s| > theta (s > theta: short basis); exit when
    s reaches the other sign or at a roll, then the entry rule is applied again on the new front contract."""
    state = np.zeros(len(feat), int)
    contract: list = [None] * len(feat)
    cur, cur_c = 0, None
    for k, (f, s) in enumerate(zip(fronts(feat), feat["spread"].to_numpy(dtype=float))):
        if f is None or not np.isfinite(s):
            cur, cur_c = 0, None
        else:
            if cur_c is not None and f != cur_c:
                cur = 0  # roll: the old position is closed
            if (cur == 1 and s <= 0) or (cur == -1 and s >= 0):
                cur = 0
            if cur == 0:
                cur = 1 if s > theta else (-1 if s < -theta else 0)
            cur_c = f if cur != 0 else None
        state[k], contract[k] = cur, cur_c
    return state, contract


def weights(index: pd.DatetimeIndex, feat: pd.DataFrame, state, contract, perp_col: str, contract_cols: list[str],
            leg: float = LEG) -> pd.DataFrame:
    """Target weights (NaN = hold) for the perpetual and its quarterlies: perp = state * leg, the held quarterly
    = -state * leg, a quarterly that is no longer held = 0. Emitted only at decisions that change something."""
    cols = [perp_col] + list(contract_cols)
    col = {c: j for j, c in enumerate(cols)}
    W = np.full((len(index), len(cols)), np.nan)
    prev_s, prev_c = 0, None
    for i, s, c in zip(feat["pos"].to_numpy(), state, contract):
        if s == prev_s and c == prev_c:
            continue
        W[i, 0] = s * leg
        if prev_c is not None and prev_c != c:
            W[i, col[prev_c]] = 0.0
        if c is not None:
            W[i, col[c]] = -s * leg
        prev_s, prev_c = s, c
    return pd.DataFrame(W, index=index, columns=cols)
