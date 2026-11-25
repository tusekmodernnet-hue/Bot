"""Round 5 (research/preregistration.md, "Runda 5"): aggressor-flow continuation (R5A), liquidation-flush fade (R5B),
sweep with aggressor absorption (R5C) and the New York opening-range breakout with a volatility filter, market or
limit execution (R5D), on BTC and ETH perps.

Market entries run through the engine on 15m bars (next-bar open, intrabar stops, funding); R5D limit entries run
through the 1-minute limit-order simulation of round 4.

Run: python -m cqs.research.run_round5 [R5A ...]
  -> research/results/R5*.json, R5*_oos.csv and R5_summary.json (Deflated Sharpe with the null variance 1/T
     pre-registered for this round, plus the cross-trial variance version). A family already in the registry is
     never rerun.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

from cqs.backtest import engine, metrics
from cqs.backtest.limit_sim import simulate
from cqs.data import store
from cqs.research import metrics_window, registry, stats
from cqs.research.run_ext import MAJORS, _fp, _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_daily
from cqs.research.run_round3 import DEV_END, checklist
from cqs.research.run_round4 import FEES, trade_metrics
from cqs.strategies import intraday as I
from cqs.strategies import orderflow as F

COLS = ["open", "high", "low", "close", "volume", "taker_buy_volume"]
H1 = pd.Timedelta(hours=1)
START = pd.Timestamp("2020-01-01", tz="UTC")
# T of the pre-registered null variance: the days behind each trial's Sharpe (the whole research period)
T_DAYS = (pd.Timestamp(DEV_END) - pd.Timestamp("2020-01-01")).days + 1
ENGINE_KEYS = ("sharpe", "cagr", "max_drawdown", "n_trades", "trades_per_year", "win_rate", "profit_factor",
               "exposure", "turnover_per_year", "costs_paid", "funding_paid")


@dataclass
class Data:
    m15: dict
    h1: dict
    oi: dict
    dclose: pd.Series
    index15: pd.DatetimeIndex
    bars: tuple
    funding: pd.DataFrame
    m1: dict = field(default_factory=dict)  # loaded only for the limit variants


def load_data() -> Data:
    m15, h1, oi = {}, {}, {}
    for s in MAJORS:
        m15[s] = store.load_klines(s, "15m").set_index("open_time")[COLS]
        h1[s] = store.load_klines(s, "1h").set_index("open_time")[COLS]
        oi[s] = store.load_metrics(s)[["create_time", "available_time", "sum_open_interest"]]
    idx = m15[MAJORS[0]].index
    for s in MAJORS[1:]:
        idx = idx.union(m15[s].index)
    bars = tuple(pd.DataFrame({s: m15[s][f] for s in MAJORS}).reindex(idx) for f in ("open", "high", "low", "close"))
    funding = pd.DataFrame({s: store.load_funding(s).set_index("funding_time")["rate"] for s in MAJORS}).sort_index()
    dclose = store.load_klines("BTCUSDT", "1d").set_index("open_time")["close"]
    return Data(m15, h1, oi, dclose, idx, bars, funding)


def load_m1(d: Data) -> None:
    for s in MAJORS:
        if s not in d.m1:
            d.m1[s] = store.load_klines(s, "1m").set_index("open_time")[["open", "high", "low", "close"]]


def fingerprint() -> str:
    """15m, 1h, 1m and 1d klines of BTC and ETH plus their open-interest snapshots."""
    oi = store.data_fingerprint([store.parquet_path("futures/um", store.METRICS, s) for s in MAJORS])
    return (_fp(MAJORS, interval="15m") + _fp(MAJORS, interval="1h") + _fp(MAJORS, interval="1m") + _fp(MAJORS)
            + oi)


# --------------------------------------------------------------------------- variant builders: (data, symbol)
def _r5a(d, s, L):
    h1 = d.h1[s]
    return I.to_15m(*F.r5a_flow(h1, L), h1.index, H1, d.m15[s].index)


def _r5b(d, s, k):
    return F.r5b_flush(d.m15[s], d.h1[s], d.oi[s], k)


def _r5c(d, s, theta):
    return F.r5c_sweep_absorption(d.m15[s], theta)


def _r5d(d, s, minutes, execution):
    if execution != "market":
        raise ValueError("limit variants are simulated on 1m bars")
    return F.r5d_orb_market(d.m15[s], d.h1[s], minutes)


GRIDS = {
    "R5A": ("r5a_flow", _r5a, [dict(L=1), dict(L=4)]),
    "R5B": ("r5b_flush", _r5b, [dict(k=3.0), dict(k=4.0)]),
    "R5C": ("r5c_sweep_absorption", _r5c, [dict(theta=0.0), dict(theta=0.1)]),
    "R5D": ("r5d_orb_ny_filtered", _r5d, [dict(minutes=m, execution=e) for m in (15, 30) for e in ("market", "limit")]),
}
assert sum(len(g[2]) for g in GRIDS.values()) == 10  # the pre-registered number of trials


def engine_variant(d: Data, fn, params: dict) -> tuple[pd.Series, pd.Series, dict, dict]:
    """Daily net returns at base and doubled costs, the per-variant summary and the full engine metrics."""
    W, S = {}, {}
    for s in MAJORS:
        w, stp = fn(d, s, **params)
        W[s] = pd.Series(np.asarray(w, float), index=d.m15[s].index).reindex(d.index15)
        S[s] = pd.Series(np.asarray(stp, float), index=d.m15[s].index).reindex(d.index15)
    W, S = pd.DataFrame(W), pd.DataFrame(S)
    res = engine.run(*d.bars, W, COSTS, stops=S, funding=d.funding, reenter_after_stop=True)
    res2 = engine.run(*d.bars, W, COSTS.scaled(2), stops=S, funding=d.funding, reenter_after_stop=True)
    m = metrics.compute(res)
    entries = int(((W != 0) & W.notna()).sum().sum())
    pv = {k: m[k] for k in ENGINE_KEYS} | {"entries": entries,
                                          "entries_per_asset": {s: int(((W[s] != 0) & W[s].notna()).sum()) for s in MAJORS}}
    if m["n_trades"] != entries:  # every decision of the strategy must reach the engine
        pv["engine_trade_mismatch"] = m["n_trades"] - entries
    r = metrics.daily_returns(res.equity, res.initial_equity)
    r2 = metrics.daily_returns(res2.equity, res2.initial_equity)
    return r, r2, pv, m


def limit_variant(d: Data, params: dict) -> tuple[pd.Series, pd.Series, dict]:
    """R5D with limit entries at the broken edge: per-contract 1m simulation, portfolio = sum of daily returns."""
    load_m1(d)
    r, r2, trades, placed = [], [], [], {}
    for s in MAJORS:
        m1 = d.m1[s]
        plans = F.r5d_orb_limit_plans(d.m15[s], d.h1[s], m1.index, params["minutes"])
        placed[s] = len(plans)
        fund = d.funding[s].dropna()
        for fees, acc in ((FEES, r), (FEES.scaled(2), r2)):
            eq, t = simulate(m1.index, m1["open"], m1["high"], m1["low"], m1["close"], plans, fees, fund)
            acc.append(metrics.daily_returns(eq))
            if fees is FEES:
                trades.append(t.assign(asset=s))
    r, r2 = sum(r).dropna(), sum(r2).dropna()
    t = pd.concat(trades, ignore_index=True)
    m = trade_metrics(r, t) | {"orders_placed": placed, "fill_rate": float(len(t) / max(sum(placed.values()), 1))}
    return r, r2, m


def run_family(fam: str, d: Data, log: bool = True) -> tuple[dict, pd.Series]:
    fname, fn, grid = GRIDS[fam]
    daily, daily_x2, per_variant = {}, {}, {}
    fp = fingerprint()
    for params in grid:
        name = _name(fname, params)
        limit = params.get("execution") == "limit"
        label = "BTC+ETH 1m limit" if limit else "BTC+ETH 15m"
        try:
            if limit:
                r, r2, m = limit_variant(d, params)
                pv = m
            else:
                r, r2, pv, m = engine_variant(d, fn, params)
        except RuntimeError as e:  # ruin is a result, not an error to hide
            per_variant[name] = {"ruined": str(e)}
            if log:
                registry.log_experiment(fam, name, params | {"market": label}, per_variant[name], fp,
                                        ("2020-01-01", DEV_END))
            continue
        daily[name], daily_x2[name], per_variant[name] = r, r2, pv
        if log:
            registry.log_experiment(fam, name, params | {"market": label}, m, fp,
                                    (str(r.index[0].date()), str(r.index[-1].date())),
                                    sharpe_per_period=stats.sharpe_per_period(r.to_numpy()))
    out = evaluate_daily(fam, daily, daily_x2, per_variant, d.dclose, "BTC+ETH", fp, log=log, keep_series=True,
                         n_variants=len(grid), start=START)
    oos = out.pop("_oos")
    out.pop("_oos_x2", None)
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


# ------------------------------------------------------------------------------------- program-level verdict
def null_sr0(n_trials: int) -> float:
    """Pre-registered (round 5) luck benchmark: expected best Sharpe of n skill-less trials whose Sharpe
    estimates have the sampling variance 1/T (per day)."""
    return stats.expected_max_sharpe(n_trials, 1.0 / T_DAYS)


def summarize(results: dict, oos: dict) -> dict:
    ex = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    n_all = len(ex)
    sr_x = [r["sharpe_per_period"] for r in ex if r.get("family", "")[:2] in ("R3", "R4", "R5")
            and r.get("family") != "R3X" and r.get("sharpe_per_period") is not None]
    var_x = float(np.var(sr_x, ddof=1))
    sr0 = null_sr0(n_all)
    fams = {}
    for fam, out in results.items():
        r = oos[fam].to_numpy()
        dsr = stats.deflated_sharpe(r, n_all, 1.0 / T_DAYS) if len(r) > 30 else float("nan")
        dsr_x = stats.deflated_sharpe(r, n_all, var_x) if len(r) > 30 else float("nan")
        checks = checklist(out, dsr)
        fams[fam] = {"oos": out["walk_forward"]["oos"], "oos_costs_x2": out["walk_forward"]["oos_costs_x2"],
                     "pbo": out["pbo"], "dsr_program_null_variance": dsr, "dsr_program_cross_trial_variance": dsr_x,
                     "regimes_oos": out["regimes_oos"], "benchmarks_same_window": out.get("benchmarks_same_window"),
                     "checks": checks, "candidate": all(checks.values())}
    n_oos = max((len(o) for o in oos.values()), default=0)
    needed = (sr0 + st.norm.ppf(0.95) / np.sqrt(n_oos - 1)) * np.sqrt(365) if n_oos > 1 else float("nan")
    return {"n_trials_program": n_all, "dsr_method": "null variance 1/T (pre-registered for round 5)",
            "T_days": T_DAYS, "sr0_annual": sr0 * np.sqrt(365),
            "oos_sharpe_for_dsr_0_95_annual_if_normal": needed, "n_oos_days": n_oos,
            "sharpe_var_rounds_3_5": var_x,
            "expected_max_sharpe_annual_cross_var": stats.expected_max_sharpe(n_all, var_x) * np.sqrt(365),
            "families": fams, "candidates": [f for f, v in fams.items() if v["candidate"]]}


def _done(fam: str) -> bool:
    """A family whose trials are already in the registry is never run (or logged) again."""
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    fams = (argv if argv is not None else sys.argv[1:]) or list(GRIDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d, ran = None, False
    for fam in fams:
        if _done(fam):
            print(fam, "already in the registry: kept, not rerun", flush=True)
            continue
        d = d or load_data()
        t0 = time.time()
        out, o = run_family(fam, d)
        o.rename(fam).to_csv(RESULTS_DIR / f"{fam}_oos.csv")
        (RESULTS_DIR / f"{fam}.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(fam, f"{time.time() - t0:.0f}s", "OOS:", out["walk_forward"]["oos"], "PBO:", out["pbo"], flush=True)
        ran = True
    if (ran or not (RESULTS_DIR / "R5_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = summarize(results, oos)
        (RESULTS_DIR / "R5_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round5_summary", {"candidates": summ["candidates"],
                                              "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
