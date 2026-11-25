"""Round 4 (research/preregistration.md, "Runda 4"): scalping with limit entries on BTC and ETH perps, simulated on
1-minute bars with conservative fill rules (`cqs.backtest.limit_sim`).

Run: python -m cqs.research.run_round4  -> research/results/R4S1.json, R4S2.json, R4*_oos.csv, R4_summary.json.
A family already in the registry is never rerun.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from cqs.backtest import metrics
from cqs.backtest.limit_sim import Fees, simulate
from cqs.data import store
from cqs.research import metrics_window, registry, stats, walkforward
from cqs.research.run_ext import MAJORS, _fp, _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_daily
from cqs.research.run_round3 import DEV_END, checklist
from cqs.strategies import scalping as S

FEES = Fees(maker=0.0002, taker=0.0005, slippage=0.0002)
GRIDS = {
    "R4S1": ("s1_vwap_limit", lambda d, s, K, trend: S.s1_plans(d["m5"][s], d["h1"][s], d["m1"][s].index, K, trend),
             [dict(K=K, trend=t) for K in (2.0, 3.0) for t in (False, True)]),
    "R4S2": ("s2_asia_limit", lambda d, s, x: S.s2_plans(d["m5"][s], d["h1"][s], d["m1"][s].index, x),
             [dict(x=x) for x in (0.0, 0.5)]),
}
assert sum(len(g[2]) for g in GRIDS.values()) == 6  # the pre-registered number of trials


def load_data() -> dict:
    d = {"m1": {}, "m5": {}, "h1": {}, "funding": {}}
    for s in MAJORS:
        for key, iv in (("m1", "1m"), ("m5", "5m"), ("h1", "1h")):
            d[key][s] = store.load_klines(s, iv).set_index("open_time")[["open", "high", "low", "close", "volume"]]
        d["funding"][s] = store.load_funding(s).set_index("funding_time")["rate"]
    d["btc_daily"] = store.load_klines("BTCUSDT", "1d").set_index("open_time")["close"]
    return d


def fingerprint() -> str:
    return _fp(MAJORS, interval="1m") + _fp(MAJORS, interval="5m") + _fp(MAJORS, interval="1h")


def run_variant(d: dict, fn, params: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Portfolio daily returns at base and doubled costs, and all trades (base costs)."""
    r, r2, trades = [], [], []
    for s in MAJORS:
        m1 = d["m1"][s]
        plans = fn(d, s, **params)
        for fees, acc in ((FEES, r), (FEES.scaled(2), r2)):
            eq, t = simulate(m1.index, m1["open"], m1["high"], m1["low"], m1["close"], plans, fees, d["funding"][s])
            acc.append(metrics.daily_returns(eq))
            if fees is FEES:
                trades.append(t.assign(asset=s))
    return sum(r).dropna(), sum(r2).dropna(), pd.concat(trades, ignore_index=True)


def trade_metrics(r: pd.Series, t: pd.DataFrame) -> dict:
    eq = (1 + r).cumprod()
    wins, losses = t.loc[t["return"] > 0, "return"], t.loc[t["return"] <= 0, "return"]
    years = len(r) / 365
    return {"sharpe": walkforward.ann_sharpe(r), "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "n_trades": int(len(t)),
            "trades_per_day": float(len(t) / max(len(r), 1)),
            "win_rate": float(len(wins) / len(t)) if len(t) else float("nan"),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("nan"),
            "avg_trade_return_on_equity": float(t["return"].mean()) if len(t) else float("nan"),
            "exit_reasons": t["reason"].value_counts().to_dict()}


def run_family(fam: str, d: dict, log: bool = True) -> tuple[dict, pd.Series]:
    fname, fn, grid = GRIDS[fam]
    daily, daily_x2, per_variant = {}, {}, {}
    fp = fingerprint()
    for params in grid:
        name = _name(fname, params)
        try:
            r, r2, t = run_variant(d, fn, params)
        except RuntimeError as e:  # ruin is a result
            per_variant[name] = {"ruined": str(e)}
            if log:
                registry.log_experiment(fam, name, params | {"market": "BTC+ETH 1m limit"}, per_variant[name], fp,
                                        ("2020-01-01", DEV_END))
            continue
        m = trade_metrics(r, t)
        daily[name], daily_x2[name], per_variant[name] = r, r2, m
        if log:
            registry.log_experiment(fam, name, params | {"market": "BTC+ETH 1m limit"}, m, fp,
                                    (str(r.index[0].date()), str(r.index[-1].date())),
                                    sharpe_per_period=stats.sharpe_per_period(r.to_numpy()))
    out = evaluate_daily(fam, daily, daily_x2, per_variant, d["btc_daily"], "BTC+ETH 1m limit", fp, log=log,
                         keep_series=True, n_variants=len(grid), start=pd.Timestamp("2020-01-01", tz="UTC"))
    oos = out.pop("_oos")
    out.pop("_oos_x2", None)
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


def summarize(results: dict, oos: dict) -> dict:
    ex = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    sr_34 = [r["sharpe_per_period"] for r in ex if r.get("family", "")[:2] in ("R3", "R4")
             and r.get("family") != "R3X" and r.get("sharpe_per_period") is not None]
    n_all, var34 = len(ex), float(np.var(sr_34, ddof=1))
    fams = {}
    for fam, out in results.items():
        r = oos[fam].to_numpy()
        dsr = stats.deflated_sharpe(r, n_all, var34) if len(r) > 30 else float("nan")
        checks = checklist(out, dsr)
        fams[fam] = {"oos": out["walk_forward"]["oos"], "oos_costs_x2": out["walk_forward"]["oos_costs_x2"],
                     "pbo": out["pbo"], "dsr_program": dsr, "regimes_oos": out["regimes_oos"],
                     "benchmarks_same_window": out.get("benchmarks_same_window"), "checks": checks,
                     "candidate": all(checks.values())}
    return {"n_trials_program": n_all, "sharpe_var_rounds_3_4": var34,
            "expected_max_sharpe_annual": stats.expected_max_sharpe(n_all, var34) * np.sqrt(365),
            "families": fams, "candidates": [f for f, v in fams.items() if v["candidate"]]}


def main(argv=None) -> int:
    fams = (argv if argv is not None else sys.argv[1:]) or list(GRIDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d = None
    for fam in fams:
        if registry.trial_count(fam) > 0:
            print(fam, "already in the registry: kept, not rerun", flush=True)
            continue
        d = d or load_data()
        t0 = time.time()
        out, o = run_family(fam, d)
        o.rename(fam).to_csv(RESULTS_DIR / f"{fam}_oos.csv")
        (RESULTS_DIR / f"{fam}.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(fam, f"{time.time() - t0:.0f}s", flush=True)
    if all(registry.trial_count(f) > 0 and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = summarize(results, oos)
        (RESULTS_DIR / "R4_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round4_summary", {"candidates": summ["candidates"], "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
