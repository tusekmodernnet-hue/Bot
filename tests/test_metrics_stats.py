import numpy as np
import pandas as pd
import pytest

from cqs.backtest import metrics
from cqs.backtest.engine import BacktestResult
from cqs.research import registry, stats


def result_from_equity(eq):
    return BacktestResult(equity=eq, positions=pd.DataFrame({"A": 0.0}, index=eq.index),
                          trades=pd.DataFrame(columns=["asset", "side", "entry_time", "exit_time", "bars", "pnl",
                                                       "return_on_equity", "exit_reason"]),
                          fills=pd.DataFrame(), costs_paid=0.0, funding_paid=0.0, traded_notional=0.0,
                          bar=pd.Timedelta("1D"))


def test_cagr_and_drawdown_known_values():
    idx = pd.date_range("2021-01-01", periods=366, freq="1D", tz="UTC")
    eq = pd.Series(np.linspace(1.0, 2.0, 366), index=idx)
    eq.iloc[100] = eq.iloc[99] * 0.8  # one-day 20% dip
    m = metrics.compute(result_from_equity(eq))
    assert m["total_return"] == pytest.approx(1.0)
    assert m["cagr"] == pytest.approx(2.0 ** (365.25 / 365) - 1, rel=1e-3)
    assert m["max_drawdown"] == pytest.approx(-0.2, abs=1e-9)


def test_drawdown_durations():
    idx = pd.date_range("2021-01-01", periods=6, freq="1D", tz="UTC")
    eq = pd.Series([1, 0.9, 0.95, 1.01, 1.0, 1.0], index=idx, dtype=float)
    d = metrics.drawdown_durations(eq)
    assert [x.days for x in d] == [3, 2]


def test_psr_is_half_at_its_own_sharpe_and_dsr_falls_with_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 1000)
    sr = stats.sharpe_per_period(r)
    assert stats.probabilistic_sharpe(r, sr) == pytest.approx(0.5)
    d10 = stats.deflated_sharpe(r, 10, 0.0005)
    d1000 = stats.deflated_sharpe(r, 1000, 0.0005)
    assert d1000 < d10
    assert stats.expected_max_sharpe(1, 0.01) == 0.0


def test_pbo_noise_vs_real_edge():
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, (1600, 20))
    pbo_noise, _ = stats.pbo_cscv(noise, n_blocks=8)
    assert 0.25 < pbo_noise < 0.75
    edge = noise.copy()
    edge[:, 0] += 0.004  # one variant with a persistent edge
    pbo_edge, _ = stats.pbo_cscv(edge, n_blocks=8)
    assert pbo_edge < 0.05


def test_registry_counts_trials_and_variance():
    for sr in (0.01, 0.03, 0.05):
        registry.log_experiment("H1", "tsmom", {"lb": 20}, {"sharpe": sr * 19}, "abc", ("2020", "2021"),
                                sharpe_per_period=sr)
    registry.log_experiment("H2", "xs", {}, {}, "abc", ("2020", "2021"), sharpe_per_period=float("nan"))
    assert registry.trial_count("H1") == 3 and registry.trial_count("H2") == 1
    assert registry.sharpe_variance("H1") == pytest.approx(np.var([0.01, 0.03, 0.05], ddof=1))
    assert registry.read()[-1]["sharpe_per_period"] is None  # NaN stored as null, not dropped


def test_pbo_skips_splits_where_no_variant_has_returns():
    rng = np.random.default_rng(4)
    m = np.zeros((160, 2))
    m[120:, 0] = rng.normal(0.001, 0.01, 40)  # both variants trade only in the last quarter
    m[120:, 1] = rng.normal(0.0, 0.01, 40)
    pbo, logits = stats.pbo_cscv(m, n_blocks=8)
    assert 0 < len(logits) < 70 and np.isfinite(pbo)  # C(8, 4) = 70 splits, those without returns in-sample skipped
