"""ML meta-model and its baseline (research/preregistration.md, entry made before the ML test).

One LightGBM classifier for BTC and ETH with fixed hyperparameters, trained in the same walk-forward windows as
the hypothesis families, with purging and embargo. The baseline is the sign of the equal-weight average of the
pre-registered directional signals. Both go through the same sizing and the same cost model.

Run: python -m cqs.research.ml_meta  (writes research/results/ML.json and logs 2 trials as family "ML")
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from cqs import config
from cqs.backtest import engine
from cqs.data import external, store
from cqs.research import panel as panel_mod
from cqs.research import registry, stats, walkforward
from cqs.research.run_ext import MAJORS, _fp, majors_panel
from cqs.research.run_family import COSTS, RESULTS_DIR, _daily_net
from cqs.strategies import breakouts
from cqs.strategies.families import MIN_CROSS_SECTION, finalize
from cqs.strategies.families_ext import daily_oi

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, bagging_fraction=0.8,
              bagging_freq=1, feature_fraction=0.8, seed=0, deterministic=True, num_threads=4, verbose=-1)
N_TREES = 200
HORIZON = 5  # label: open t+1 -> open t+6
EMBARGO_DAYS = 5
VOL_WARMUP_DAYS = 60  # the portfolio vol target needs 60 days of the strategy's own positions
RETURN_LAGS = (1, 3, 20, 40, 60, 90, 120)
TREND_LAGS = (20, 40, 60, 90, 120)


def _ns(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).as_unit("ns")


def _xs_pct(score: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each contract's score within that day's universe (NaN below the minimum cross-section)."""
    s = score.where(universe)
    enough = s.notna().sum(axis=1) >= MIN_CROSS_SECTION
    return s.rank(axis=1, pct=True).where(enough, axis=0)


def h6_daily(days: pd.DatetimeIndex, symbols=MAJORS) -> pd.DataFrame:
    """Net direction of H6 (N=24, M=1) entries decided during each UTC day; the day's last hourly bar closes
    at the daily close, so the value is known then."""
    out = {}
    for a in symbols:
        k = store.load_klines(a, "1h").set_index("open_time")
        w, _ = breakouts.false_breakout(k["open"], k["high"], k["low"], k["close"], 24, 1)
        e = np.sign(w.where(w.notna() & (w != 0))).fillna(0.0)
        d = e.groupby(_ns(e.index.normalize())).sum()
        out[a] = d
    return pd.DataFrame(out).reindex(_ns(days)).set_axis(days)


def externals(days: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Off-exchange series, already shifted to the day they become usable, carried forward over gaps."""
    out = {}
    for name, s in (("fear_greed", external.fear_greed()), ("mvrv", external.coinmetrics()),
                    ("vix", external.fred("VIXCLS"))):
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = _ns(s.index)
        out[name] = s.reindex(_ns(days), method="ffill").set_axis(days)
    return out


def features(pm: panel_mod.Panel, pu: panel_mod.Panel, oi: pd.DataFrame | None = None,
             h6: pd.DataFrame | None = None, ext: dict[str, pd.Series] | None = None) -> dict[str, pd.DataFrame]:
    """Feature frames (day x major), each value known at that day's close. `pu` is the top-50 universe panel."""
    c = pm.close
    f: dict[str, pd.DataFrame] = {}
    for L in RETURN_LAGS:
        f[f"ret_{L}"] = c / c.shift(L) - 1.0
    mom = pu.close.shift(1) / pu.close.shift(31) - 1.0  # H2a score, last day skipped
    f["xs_mom30_pct"] = _xs_pct(mom, pu.universe).reindex(index=c.index, columns=c.columns)
    fund = pm.daily_funding()
    for W in (1, 3, 7, 14):
        f[f"funding_{W}d"] = fund.rolling(W, min_periods=W).sum()
    f["funding_z30"] = (fund - fund.rolling(30, min_periods=30).mean()) / fund.rolling(30, min_periods=30).std()
    fu = pu.daily_funding().rolling(14, min_periods=14).sum()
    f["xs_funding14_pct"] = _xs_pct(fu, pu.universe).reindex(index=c.index, columns=c.columns)
    if oi is not None:
        f["oi_chg_3d"] = oi / oi.shift(3) - 1.0
    f["vol_30"] = c.pct_change(fill_method=None).rolling(30, min_periods=30).std() * np.sqrt(365)
    if h6 is not None:
        f["h6_entries"] = h6.reindex(index=c.index, columns=c.columns)
    for name, s in (ext or {}).items():
        f[name] = pd.DataFrame({a: s.reindex(c.index) for a in c.columns})
    return f


def labels(pm: panel_mod.Panel) -> pd.DataFrame:
    o = pm.open
    fwd = o.shift(-(HORIZON + 1)) / o.shift(-1) - 1.0
    return (fwd > 0).astype(float).where(fwd.notna())


def to_long(f: dict[str, pd.DataFrame], y: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    X = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    X["label"] = y.stack(future_stack=True)
    live = close.notna().stack(future_stack=True)
    return X[live.reindex(X.index).fillna(False).to_numpy()]


def train_mask(dates: pd.DatetimeIndex, train_start, test_start) -> np.ndarray:
    """Purging + embargo: a training day t is kept only if its label (known at the open of t+6) is known
    before the test starts, with EMBARGO_DAYS more days of gap."""
    last = test_start - pd.Timedelta(days=HORIZON + 1 + EMBARGO_DAYS)
    return np.asarray((dates >= train_start) & (dates <= last))


def predict_oos(X: pd.DataFrame, feats: list[str], start) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward predictions (probability of an up move) and gain importance per window."""
    import lightgbm as lgb

    dates = X.index.get_level_values(0)
    days = pd.DatetimeIndex(sorted(dates.unique()))
    preds, imps = [], {}
    for tr0, _tr1, te0, te1 in walkforward.windows(days, start=start):
        tr = X[train_mask(dates, tr0, te0) & X["label"].notna().to_numpy()]
        te = X[np.asarray((dates >= te0) & (dates < te1))]
        if te.empty or len(tr) < 200:
            continue
        model = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr["label"]), num_boost_round=N_TREES)
        preds.append(pd.Series(model.predict(te[feats]), index=te.index))
        imps[str(te0.date())] = pd.Series(model.feature_importance("gain"), index=feats)
    return pd.concat(preds), pd.DataFrame(imps)


def baseline_signal(f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sign of the equal-weight average of six pre-registered directions (H1, H2a, H2b, H3a, H3b, H6)."""
    h1 = sum(np.sign(f[f"ret_{L}"]) for L in TREND_LAGS) / len(TREND_LAGS)
    pm_ = f["xs_mom30_pct"]
    h2a = ((pm_ > 0.8).astype(float) - (pm_ <= 0.2).astype(float)).where(pm_.notna(), 0.0)
    h2b = -np.sign(f["ret_3"])
    pf = f["xs_funding14_pct"]
    h3a = ((pf <= 0.2).astype(float) - (pf > 0.8).astype(float)).where(pf.notna(), 0.0)
    z = f["funding_z30"]
    h3b = (-np.sign(z)).where(z.abs() > 1.5, 0.0)
    comps = [h1, h2a, h2b, h3a, h3b]
    if "h6_entries" in f:
        comps.append(np.sign(f["h6_entries"]))
    return np.sign(sum(x.fillna(0.0) for x in comps) / len(comps))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    r = pd.Series(p).rank().to_numpy()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _evaluate(name, params, sig, pm, oos_start, n_trials, var_sr, fingerprint, reg, log):
    w = finalize(sig, pm)
    w.loc[w.index < oos_start] = 0.0  # vol-target warm-up: flat, and not evaluated
    bars = (pm.open, pm.high, pm.low, pm.close)
    res = engine.run(*bars, w, COSTS, funding=pm.funding, rebalance_band=0.02)
    res2 = engine.run(*bars, w, COSTS.scaled(2), funding=pm.funding, rebalance_band=0.02)
    r = _daily_net(res)
    r = r[r.index >= oos_start]
    r2 = _daily_net(res2)
    r2 = r2[r2.index >= oos_start]
    summ = walkforward.summary(r)
    dsr = stats.deflated_sharpe(r.to_numpy(), n_trials, var_sr)
    out = {"oos": summ, "oos_costs_x2": walkforward.summary(r2), "deflated_sharpe": dsr,
           "regimes_oos": walkforward.by_regime(r, reg), "n_trades": len(res.trades),
           "costs_paid": float(res.costs_paid), "funding_paid": float(res.funding_paid)}
    if log:
        registry.log_experiment("ML", name, params, summ | {"deflated_sharpe": dsr}, fingerprint,
                                (str(r.index[0].date()), str(r.index[-1].date())),
                                sharpe_per_period=stats.sharpe_per_period(r.to_numpy()))
    return out


def run(log: bool = True) -> dict:
    pm = majors_panel()
    syms = open(config.DATA_DIR / "universe_ever_top50.txt").read().split()
    pu = panel_mod.load(syms)
    days = pm.close.index
    oi = pd.DataFrame({a: daily_oi(a, days) for a in MAJORS})
    f = features(pm, pu, oi=oi, h6=h6_daily(days), ext=externals(days))
    feats = list(f)
    X = to_long(f, labels(pm), pm.close)
    start = pd.Timestamp(config.DEV_START)
    p, imp = predict_oos(X, feats, start)
    first_test = p.index.get_level_values(0).min()
    oos_start = first_test + pd.Timedelta(days=VOL_WARMUP_DAYS)

    prob = p.unstack().reindex(index=days, columns=pm.close.columns)
    sig_ml = np.sign(prob - 0.5)  # NaN (no view) before the first test window
    # same start as the model, so both strategies have the same vol-target warm-up
    sig_base = baseline_signal(f).where(pd.Series(days >= first_test, index=days), np.nan, axis=0)

    experiments = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    n_trials = len(experiments) + 2
    srs = [r["sharpe_per_period"] for r in experiments if r.get("sharpe_per_period") is not None]
    var_sr = float(np.var(srs, ddof=1))
    reg = walkforward.regimes(pm.close["BTCUSDT"])
    fp = _fp(MAJORS) + _fp(syms)
    y = X.loc[p.index, "label"]
    known = y.notna().to_numpy()
    pred_stats = {"rows": int(len(p)), "accuracy": float(((p.to_numpy()[known] > 0.5) == (y.to_numpy()[known] == 1)).mean()),
                  "auc": _auc(p.to_numpy()[known], y.to_numpy()[known]),
                  "base_rate_up": float(y[known].mean()), "share_long": float((p > 0.5).mean())}
    out = {
        "family": "ML", "market": "BTC+ETH", "data": fp, "features": feats,
        "oos_start": str(oos_start.date()), "first_test": str(first_test.date()),
        "n_trials_registry": n_trials, "sharpe_variance_registry": var_sr,
        "prediction": pred_stats,
        "importance_gain_mean": imp.mean(axis=1).sort_values(ascending=False).to_dict(),
        "model": _evaluate("lgbm_meta(fixed)", {"params": PARAMS, "trees": N_TREES, "horizon": HORIZON,
                                                "embargo": EMBARGO_DAYS}, sig_ml, pm, oos_start, n_trials,
                           var_sr, fp, reg, log),
        "baseline": _evaluate("signal_combo_equal(H1,H2a,H2b,H3a,H3b,H6)", {"combo": "equal-weight sign"},
                              sig_base, pm, oos_start, n_trials, var_sr, fp, reg, log),
    }
    if log:
        registry.log_event("family_evaluated", {"family": "ML", "market": "BTC+ETH",
                                                "oos_model": out["model"]["oos"],
                                                "oos_baseline": out["baseline"]["oos"]})
    return out


def main() -> int:
    t = time.time()
    out = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "ML.json"
    path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
    print(path, f"{time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
