"""Run one pre-registered family end to end: every variant, registry logging, PBO, walk-forward OOS,
Deflated Sharpe, regime breakdown and the 2x cost stress test. Writes research/results/<family>.json."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from cqs import config
from cqs.backtest import engine, metrics
from cqs.backtest.costs import CostModel
from cqs.data import store
from cqs.research import panel as panel_mod
from cqs.research import registry, stats, walkforward
from cqs.strategies import families

COSTS = CostModel(default_slippage_bps=10.0, slippage_bps={"BTCUSDT": 2.0, "ETHUSDT": 2.0})
RESULTS_DIR = config.REPO_ROOT / "research" / "results"


def _daily_net(res) -> pd.Series:
    return metrics.daily_returns(res.equity, res.initial_equity)


def evaluate(family: str, p: panel_mod.Panel, market_label: str, fingerprint: str, log: bool = True) -> dict:
    variants = families.GRIDS[family]
    funding = p.funding
    daily, daily_x2, per_variant = {}, {}, {}
    for fname, params in variants:
        name = f"{fname}({', '.join(f'{k}={v}' for k, v in params.items())})"
        w = families.FUNCS[fname](p, **params)
        try:
            res = engine.run(p.open, p.high, p.low, p.close, w, COSTS, funding=funding, rebalance_band=0.02)
            res2 = engine.run(p.open, p.high, p.low, p.close, w, COSTS.scaled(2), funding=funding, rebalance_band=0.02)
        except RuntimeError as e:  # ruin is a result, not an error to hide
            m = {"ruined": str(e)}
            if log:
                registry.log_experiment(family, name, params | {"market": market_label}, m, fingerprint,
                                        (str(p.close.index[0].date()), str(p.close.index[-1].date())))
            per_variant[name] = m
            continue
        r = _daily_net(res)
        m = metrics.compute(res)
        daily[name], daily_x2[name] = r, _daily_net(res2)
        per_variant[name] = {k: m[k] for k in ("sharpe", "cagr", "max_drawdown", "n_trades", "win_rate",
                                                "profit_factor", "exposure", "turnover_per_year",
                                                "costs_paid", "funding_paid")}
        if log:
            registry.log_experiment(family, name, params | {"market": market_label}, m, fingerprint,
                                    (str(p.close.index[0].date()), str(p.close.index[-1].date())),
                                    sharpe_per_period=stats.sharpe_per_period(r.to_numpy()))
    D = pd.DataFrame(daily).dropna(how="all").fillna(0.0)
    D2 = pd.DataFrame(daily_x2).reindex_like(D).fillna(0.0)
    # warm-up: drop leading days where no variant has any return
    active = (D != 0).any(axis=1)
    D, D2 = D.loc[active.idxmax():], D2.loc[active.idxmax():]
    oos, choices = walkforward.select_oos(D, start=p.close.index[0])
    oos2 = pd.concat([D2.loc[(D2.index >= c["test_start"]) & (D2.index < c["test_end"]), c["variant"]]
                      for c in choices]) if choices else pd.Series(dtype=float)
    pbo, _ = stats.pbo_cscv(D.to_numpy(), n_blocks=16) if D.shape[1] >= 2 else (float("nan"), None)
    n_trials = registry.trial_count(family) if log else len(variants)
    var_sr = registry.sharpe_variance(family) if log else float(np.var([stats.sharpe_per_period(D[c]) for c in D], ddof=1))
    dsr = stats.deflated_sharpe(oos.to_numpy(), n_trials, var_sr) if len(oos) > 30 else float("nan")
    reg = walkforward.regimes(p.close["BTCUSDT"]) if "BTCUSDT" in p.close else None
    out = {
        "family": family,
        "market": market_label,
        "data": fingerprint,
        "variants": per_variant,
        "walk_forward": {"oos": walkforward.summary(oos) if len(oos) else None,
                         "oos_costs_x2": walkforward.summary(oos2) if len(oos2) else None,
                         "choices": choices},
        "pbo": pbo,
        "deflated_sharpe": dsr,
        "n_trials_family": n_trials,
        "regimes_oos": walkforward.by_regime(oos, reg) if reg is not None and len(oos) else None,
    }
    if log:
        registry.log_event("family_evaluated", {"family": family, "market": market_label, "pbo": pbo,
                                                "deflated_sharpe": dsr, "oos": out["walk_forward"]["oos"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(families.GRIDS))
    ap.add_argument("--symbols-file", default=str(config.DATA_DIR / "universe_ever_top50.txt"))
    ap.add_argument("--majors-only", action="store_true", help="BTC and ETH only")
    a = ap.parse_args(argv)
    t = time.time()
    syms = ["BTCUSDT", "ETHUSDT"] if a.majors_only else open(a.symbols_file).read().split()
    p = panel_mod.load(syms)
    if a.majors_only:
        p.universe = p.close.notna()  # BTC and ETH are always eligible while they trade
    label = "BTC+ETH" if a.majors_only else f"universe_top50({len(syms)})"
    fp = store.data_fingerprint([store.parquet_path("futures/um", store.KLINES, s, "1d") for s in p.close.columns])
    out = evaluate(a.family, p, label, fp)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "majors" if a.majors_only else "universe"
    path = RESULTS_DIR / f"{a.family}_{tag}.json"
    path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
    print(path, f"{time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
