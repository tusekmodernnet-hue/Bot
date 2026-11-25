"""Statistics that separate a real edge from the best of many lucky trials.

- Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): PSR against the Sharpe ratio expected from the
  best of N independent trials with no skill.
- Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation
  (Bailey, Borwein, Lopez de Prado & Zhu, 2017).
All Sharpe ratios here are per period (not annualised), matching the return frequency passed in.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_per_period(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given the sample, its skewness and kurtosis."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    sr = sharpe_per_period(r)
    g3 = stats.skew(r, bias=False)
    g4 = stats.kurtosis(r, fisher=False, bias=False)
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe among n_trials skill-less trials whose Sharpe ratios have this variance."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(r: np.ndarray, n_trials: int, sr_variance: float) -> float:
    """Probability that the strategy's Sharpe beats the best-of-N luck benchmark. Protocol threshold: 0.95."""
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, sr_variance))


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16) -> tuple[float, np.ndarray]:
    """Probability of backtest overfitting.

    returns: T x N matrix, one column per strategy variant tried. The rows are split into n_blocks
    contiguous blocks; for every half/half split the variant with the best in-sample Sharpe is located in
    the out-of-sample ranking. PBO is the share of splits where it lands in the bottom half.
    Protocol threshold: PBO <= 0.25.
    """
    m = np.asarray(returns, dtype=float)
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least two strategy variants")
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be even and >= 4")
    blocks = np.array_split(np.arange(t), n_blocks)
    logits = []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[k] for k in combo])
        oos_idx = np.concatenate([blocks[k] for k in range(n_blocks) if k not in combo])
        sr_is = np.array([sharpe_per_period(m[is_idx, j]) for j in range(n)])
        sr_oos = np.array([sharpe_per_period(m[oos_idx, j]) for j in range(n)])
        if np.isnan(sr_is).all():
            continue  # no variant has returns in this half: there is no in-sample winner to locate
        best = int(np.nanargmax(sr_is))
        # relative rank of the IS winner out of sample, in (0, 1)
        rank = (stats.rankdata(sr_oos)[best]) / (n + 1)
        logits.append(np.log(rank / (1.0 - rank)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits
