import numpy as np
import pandas as pd
import pytest

from cqs.data.universe import UniverseRules, eligible
from cqs.research.panel import Panel
from cqs.strategies import families


def synthetic_panel(days=420, n=24, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=days, freq="1D", tz="UTC")
    cols = ["BTCUSDT", "ETHUSDT"] + [f"A{i}USDT" for i in range(n - 2)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (days, n)), axis=0)), index=idx, columns=cols)
    close.iloc[:150, -1] = np.nan  # a late listing
    close.iloc[300:, -2] = np.nan  # a delisting
    open_ = close.shift(1).fillna(close)
    qv = close.notna() * 1e8
    vol = close.notna() * 1.0
    uni = eligible({"close": close, "quote_volume": qv.where(close.notna()), "volume": vol.where(close.notna())},
                   UniverseRules(min_age_days=30, top_n=None, min_avg_quote_volume=1))
    fidx = pd.date_range("2021-01-01", periods=days * 3, freq="8h", tz="UTC")
    funding = pd.DataFrame(rng.normal(1e-4, 2e-4, (len(fidx), n)), index=fidx, columns=cols)
    return Panel(open_, np.maximum(open_, close), np.minimum(open_, close), close, qv, funding, uni)


def cut_future(p: Panel, t: pd.Timestamp) -> Panel:
    """Same panel, but every value after t is replaced (prices tripled, funding flipped)."""
    c = p.close.copy()
    c.loc[c.index > t] *= 3.0
    f = p.funding.copy()
    f.loc[f.index > t + pd.Timedelta(days=1)] *= -5
    return Panel(p.open, p.high, p.low, c, p.quote_volume, f, p.universe)


@pytest.mark.parametrize("family", list(families.GRIDS))
def test_family_weights_do_not_use_future_data(family):
    p = synthetic_panel()
    t = p.close.index[350]
    fname, params = families.GRIDS[family][0]
    w1 = families.FUNCS[fname](p, **params)
    w2 = families.FUNCS[fname](cut_future(p, t), **params)
    pd.testing.assert_frame_equal(w1.loc[:t], w2.loc[:t])


@pytest.mark.parametrize("family", list(families.GRIDS))
def test_family_weights_respect_caps_and_data(family):
    p = synthetic_panel()
    for fname, params in families.GRIDS[family]:
        w = families.FUNCS[fname](p, **params)
        assert (w.fillna(0).abs().sum(axis=1) <= 2.0 + 1e-9).all()
        alts = [c for c in w.columns if c not in ("BTCUSDT", "ETHUSDT")]
        assert (w[alts].fillna(0).abs() <= 0.25 + 1e-12).all().all()
        assert (w.fillna(0).where(p.close.isna(), 0.0) == 0).all().all()
        assert w.fillna(0).abs().sum().sum() > 0  # the variant actually trades
