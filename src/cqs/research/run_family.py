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
    """Daily families: weights from the pre-registered grid, simulated on daily bars."""
    runs = []
    for fname, params in families.GRIDS[family]:
        name = f"{fname}({', '.join(f'{k}={v}' for k, v in params.items())})"
        runs.append((name, params, families.FUNCS[fname](p, **params), None))
    bars = (p.open, p.high, p.low, p.close)
    return evaluate_runs(family, runs, bars, p.funding, p.close["BTCUSDT"] if "BTCUSDT" in p.close else None,
                         market_label, fingerprint, log, rebalance_band=0.02)


def program_trials() -> tuple[int, float]:
    """Number of strategy trials in the whole registry (benchmarks excluded) and the variance of their
    per-period Sharpe ratios: the multiple-testing correction for round 2."""
    ex = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    srs = [r["sharpe_per_period"] for r in ex if r.get("sharpe_per_period") is not None]
    return len(ex), float(np.var(srs, ddof=1)) if len(srs) > 1 else float("nan")


def evaluate_runs(family, runs, bars, funding, btc_daily_close, market_label, fingerprint, log=True,
                  rebalance_band=0.0, costs: CostModel = COSTS, return_adjust=None, wf_start=None,
                  keep_series: bool = False, engine_kwargs: dict | None = None) -> dict:
    """`return_adjust(res) -> daily series` is subtracted from each run's daily returns (R2A: the T-bill yield
    on the capital tied up). `wf_start` overrides the first walk-forward window start. With `keep_series`, the
    stitched out-of-sample daily returns are returned under "_oos" (not JSON: the caller removes it)."""
    o, h, l, c = bars
    period = (str(c.index[0].date()), str(c.index[-1].date()))
    daily, daily_x2, per_variant = {}, {}, {}
    variants = runs
    for name, params, w, stops in runs:
        try:
            ek = engine_kwargs or {}
            res = engine.run(o, h, l, c, w, costs, stops=stops, funding=funding, rebalance_band=rebalance_band, **ek)
            res2 = engine.run(o, h, l, c, w, costs.scaled(2), stops=stops, funding=funding,
                              rebalance_band=rebalance_band, **ek)
        except RuntimeError as e:  # ruin is a result, not an error to hide
            m = {"ruined": str(e)}
            if log:
                registry.log_experiment(family, name, params | {"market": market_label}, m, fingerprint, period)
            per_variant[name] = m
            continue
        r, r_x2 = _daily_net(res), _daily_net(res2)
        if return_adjust is not None:
            r = r - return_adjust(res).reindex(r.index).fillna(0.0)
            r_x2 = r_x2 - return_adjust(res2).reindex(r_x2.index).fillna(0.0)
        m = metrics.compute(res)
        if return_adjust is not None:
            m["sharpe_excess"] = walkforward.ann_sharpe(r)
            m["total_excess_return"] = float((1 + r).prod() - 1)
        daily[name], daily_x2[name] = r, r_x2
        per_variant[name] = {k: m[k] for k in ("sharpe", "cagr", "max_drawdown", "n_trades", "win_rate",
                                                "profit_factor", "exposure", "turnover_per_year",
                                                "costs_paid", "funding_paid")}
        if log:
            registry.log_experiment(family, name, params | {"market": market_label}, m, fingerprint, period,
                                    sharpe_per_period=stats.sharpe_per_period(r.to_numpy()))
    return evaluate_daily(family, daily, daily_x2, per_variant, btc_daily_close, market_label, fingerprint, log,
                          wf_start, keep_series, n_variants=len(variants), start=c.index[0].normalize())


def evaluate_h6(symbols: list[str], log: bool = True) -> dict:
    """H6 on hourly bars, with intrabar stops."""
    from cqs.research.benchmarks import load_panel
    from cqs.strategies import breakouts
    o, h, l, c, funding = load_panel(symbols, "1h")
    runs = []
    for params in breakouts.GRID:
        w, st = breakouts.h6_weights(o, h, l, c, **params)
        runs.append((f"h6_false_breakout(n={params['n']}, m={params['m']})", params, w, st))
    daily_btc = c["BTCUSDT"].groupby(c.index.normalize()).last() if "BTCUSDT" in c else None
    fp = store.data_fingerprint([store.parquet_path("futures/um", store.KLINES, s, "1h") for s in symbols])
    return evaluate_runs("H6", runs, (o, h, l, c), funding, daily_btc, "+".join(symbols), fp, log)


def h6_entry_gate(index: pd.DatetimeIndex, top_n: int) -> pd.DataFrame:
    """Bool frame (hourly bars x contracts ever in the point-in-time top N): True where a new trade may open.

    Membership at the close of day d allows entries from day d+1 on, so the gate uses no future data.
    """
    from cqs.data.universe import UniverseRules, daily_panel, eligible
    syms = open(config.DATA_DIR / "universe_ever_top50.txt").read().split()
    el = eligible(daily_panel(syms), UniverseRules(top_n=top_n))
    el = el.loc[:, el.any()]
    el.index = pd.DatetimeIndex(el.index).as_unit("ns") + pd.Timedelta(days=1)
    gate = el.reindex(pd.DatetimeIndex(index.normalize()).as_unit("ns")).fillna(False).astype(bool)
    gate.index = index
    return gate


def evaluate_h6_top(top_n: int = 20, log: bool = True) -> dict:
    """H6 on the point-in-time top-N universe, with the common position caps and the 2x gross cap."""
    from cqs.research.benchmarks import load_panel
    from cqs.strategies import breakouts
    from cqs.data.universe import UniverseRules, daily_panel, eligible
    all_syms = open(config.DATA_DIR / "universe_ever_top50.txt").read().split()
    el = eligible(daily_panel(all_syms), UniverseRules(top_n=top_n))
    symbols = sorted(el.columns[el.any()])
    o, h, l, c, funding = load_panel(symbols, "1h")
    missing = sorted(set(c.columns) - set(funding.columns))
    if missing:  # funding is a real cost; never run without it
        raise SystemExit(f"funding data missing for {len(missing)} symbols, e.g. {missing[:5]}")
    gate = h6_entry_gate(c.index, top_n).reindex(columns=c.columns, fill_value=False)
    caps = {s: (1.0 if s in ("BTCUSDT", "ETHUSDT") else 0.25) for s in c.columns}
    runs = []
    for params in breakouts.GRID:
        w, st = breakouts.h6_weights(o, h, l, c, **params, can_enter=gate, max_weight=caps, max_gross=2.0)
        runs.append((f"h6_false_breakout(n={params['n']}, m={params['m']}, universe=top{top_n})",
                     params | {"universe": f"top{top_n}"}, w, st))
    daily_btc = c["BTCUSDT"].groupby(c.index.normalize()).last()
    fp = store.data_fingerprint([store.parquet_path("futures/um", store.KLINES, s, "1h") for s in symbols]
                                + [store.parquet_path("futures/um", store.FUNDING, s) for s in symbols])
    return evaluate_runs("H6", runs, (o, h, l, c), funding, daily_btc, f"universe_top{top_n}({len(symbols)})",
                         fp, log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(families.GRIDS) + ["H6"])
    ap.add_argument("--symbols-file", default=str(config.DATA_DIR / "universe_ever_top50.txt"))
    ap.add_argument("--majors-only", action="store_true", help="BTC and ETH only")
    ap.add_argument("--top", type=int, default=None, help="H6 only: point-in-time top-N universe")
    a = ap.parse_args(argv)
    t = time.time()
    if a.family == "H6" and a.top:
        out = evaluate_h6_top(a.top)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"H6_top{a.top}.json"
        path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(path, f"{time.time() - t:.0f}s")
        return 0
    if a.family == "H6":
        syms = ["BTCUSDT", "ETHUSDT"] if a.majors_only else open(a.symbols_file).read().split()
        out = evaluate_h6(syms)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"H6_{'majors' if a.majors_only else 'top'}.json"
        path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(path, f"{time.time() - t:.0f}s")
        return 0
    syms = ["BTCUSDT", "ETHUSDT"] if a.majors_only else open(a.symbols_file).read().split()
    p = panel_mod.load(syms)
    if a.majors_only:
        p.universe = p.close.notna()  # BTC and ETH are always eligible while they trade
    missing = sorted(set(p.close.columns) - set(p.funding.columns))
    if missing:  # funding is a real cost; never run without it
        raise SystemExit(f"funding data missing for {len(missing)} symbols, e.g. {missing[:5]}")
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


def evaluate_daily(family, daily: dict, daily_x2: dict, per_variant: dict, btc_daily_close, market_label, fingerprint,
                   log=True, wf_start=None, keep_series: bool = False, n_variants: int | None = None,
                   start=None) -> dict:
    """Walk-forward, PBO, Deflated Sharpe, regimes and the 2x cost stress from each variant's daily net returns
    (base and doubled costs). Trials must already be logged by the caller."""
    D = pd.DataFrame(daily).dropna(how="all").fillna(0.0)
    D2 = pd.DataFrame(daily_x2).reindex_like(D).fillna(0.0)
    # warm-up: drop leading days where no variant has any return
    active = (D != 0).any(axis=1)
    D, D2 = D.loc[active.idxmax():], D2.loc[active.idxmax():]
    oos, choices = walkforward.select_oos(D, start=wf_start if wf_start is not None else start)
    oos2 = pd.concat([D2.loc[(D2.index >= ch["test_start"]) & (D2.index < ch["test_end"]), ch["variant"]]
                      for ch in choices]) if choices else pd.Series(dtype=float)
    pbo, _ = stats.pbo_cscv(D.to_numpy(), n_blocks=16) if D.shape[1] >= 2 else (float("nan"), None)
    n_trials = registry.trial_count(family) if log else n_variants
    var_sr = registry.sharpe_variance(family) if log else float(np.var([stats.sharpe_per_period(D[c]) for c in D], ddof=1))
    dsr = stats.deflated_sharpe(oos.to_numpy(), n_trials, var_sr) if len(oos) > 30 else float("nan")
    reg = walkforward.regimes(btc_daily_close) if btc_daily_close is not None else None
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
        "deflated_sharpe_program": (stats.deflated_sharpe(oos.to_numpy(), *program_trials())
                                    if log and len(oos) > 30 else float("nan")),
        "n_trials_program": program_trials()[0] if log else None,
        "n_trials_family": n_trials,
        "regimes_oos": walkforward.by_regime(oos, reg) if reg is not None and len(oos) else None,
    }
    if keep_series:
        out["_oos"], out["_oos_x2"] = oos, oos2
    if log:
        registry.log_event("family_evaluated", {"family": family, "market": market_label, "pbo": pbo,
                                                "deflated_sharpe": dsr, "oos": out["walk_forward"]["oos"]})
    return out
