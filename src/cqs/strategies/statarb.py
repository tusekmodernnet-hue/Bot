"""R7B (research/preregistration.md, "Runda 7"): residual mean reversion against BTC, after Avellaneda and Lee (2010).

On each day t and for each coin of the point-in-time top-20 universe (BTC excluded, it is the hedge), the last 60
daily log returns are regressed on BTC's: r = alpha + beta * r_btc + eps. X is the running sum of the residuals over
the window, and an AR(1) fit X[k+1] = a + b * X[k] + zeta gives the mean-reversion speed. The model is valid when
0 < b < exp(-1/30) (mean-reversion time under 30 days); then m = a / (1 - b), sigma_eq = sqrt(var(zeta) / (1 - b^2))
and the s-score is (X[-1] - m) / sigma_eq. Since OLS residuals sum to zero over the window, X[-1] is 0 and the score
is -m / sigma_eq, as in the paper.

Positions: s < -s_in opens a long in the coin, s > s_in a short; a long closes when s > -0.5 and a short when
s < 0.75 (the paper's thresholds), or when the coin leaves the universe or its model stops being valid. Each
position is 0.1 of equity in the coin, hedged with beta * 0.1 in BTC (beta clipped to [0, 3], re-estimated daily);
at most 10 positions, larger |s| first. A closed position may reopen on the other side at the same decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 60
B_MAX = float(np.exp(-1.0 / 30.0))
CLOSE_LONG = -0.5
CLOSE_SHORT = 0.75
SIZE = 0.1
MAX_POS = 10
BETA_CLIP = (0.0, 3.0)
HEDGE = "BTCUSDT"


def scores(logret: pd.DataFrame, btc: pd.Series, universe: pd.DataFrame,
           window: int = WINDOW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """s-score, beta and model validity (day x coin), computed only where the coin is in the universe and the last
    `window` returns of both the coin and BTC exist. Row t uses returns up to day t only."""
    idx = logret.index
    x_all = btc.reindex(idx).to_numpy(dtype=float)
    U = universe.reindex(index=idx, columns=logret.columns).fillna(False).to_numpy(dtype=bool)
    S = np.full(logret.shape, np.nan)
    B = np.full(logret.shape, np.nan)
    V = np.zeros(logret.shape, dtype=bool)
    for j, c in enumerate(logret.columns):
        y_all = logret[c].to_numpy(dtype=float)
        for i in np.flatnonzero(U[:, j]):
            if i < window - 1:
                continue
            y = y_all[i - window + 1:i + 1]
            x = x_all[i - window + 1:i + 1]
            if np.isnan(y).any() or np.isnan(x).any():
                continue
            xd = x - x.mean()
            vx = float(xd @ xd)
            if vx <= 0:
                continue
            beta = float(xd @ (y - y.mean())) / vx
            alpha = y.mean() - beta * x.mean()
            X = np.cumsum(y - alpha - beta * x)
            x0, x1 = X[:-1], X[1:]
            d0 = x0 - x0.mean()
            v0 = float(d0 @ d0)
            B[i, j] = beta
            if v0 <= 0:
                continue
            b = float(d0 @ (x1 - x1.mean())) / v0
            if not (0.0 < b < B_MAX):
                continue
            a = x1.mean() - b * x0.mean()
            zeta = x1 - a - b * x0
            sig = np.sqrt(zeta.var(ddof=1) / (1.0 - b * b))
            if not sig > 0:
                continue
            S[i, j] = (X[-1] - a / (1.0 - b)) / sig
            V[i, j] = True
    mk = lambda a: pd.DataFrame(a, index=idx, columns=logret.columns)  # noqa: E731
    return mk(S), mk(B), mk(V)


def positions(s: pd.DataFrame, valid: pd.DataFrame, universe: pd.DataFrame, s_in: float,
              max_pos: int = MAX_POS) -> pd.DataFrame:
    """Position sign (+1 long, -1 short, 0 flat) per day and coin after each decision."""
    cols = list(s.columns)
    S = s.to_numpy(dtype=float)
    V = valid.reindex_like(s).fillna(False).to_numpy(dtype=bool)
    U = universe.reindex_like(s).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(S.shape)
    pos: dict[int, int] = {}
    for i in range(S.shape[0]):
        for j, sg in list(pos.items()):
            v = S[i, j]
            if not U[i, j] or not V[i, j] or np.isnan(v) or (sg > 0 and v > CLOSE_LONG) or (sg < 0 and v < CLOSE_SHORT):
                del pos[j]
        free = max_pos - len(pos)
        if free > 0:
            cand = []
            for j in range(len(cols)):
                v = S[i, j]
                if j in pos or not U[i, j] or not V[i, j] or np.isnan(v):
                    continue
                if v < -s_in:
                    cand.append((abs(v), j, 1))
                elif v > s_in:
                    cand.append((abs(v), j, -1))
            cand.sort(key=lambda t: (-t[0], cols[t[1]]))
            for _, j, sg in cand[:free]:
                pos[j] = sg
        for j, sg in pos.items():
            out[i, j] = sg
    return pd.DataFrame(out, index=s.index, columns=cols)


def weights(sign: pd.DataFrame, beta: pd.DataFrame, size: float = SIZE, hedge: str = HEDGE) -> pd.DataFrame:
    """Target weights: sign * size in each coin and the summed beta hedge in BTC."""
    w = sign * size
    b = beta.reindex_like(sign).clip(*BETA_CLIP).fillna(0.0)
    w[hedge] = -(sign * b * size).sum(axis=1)
    return w


def r7b(logret: pd.DataFrame, btc: pd.Series, universe: pd.DataFrame, s_in: float):
    s, b, v = scores(logret, btc, universe)
    sign = positions(s, v, universe, s_in)
    return weights(sign, b), sign
