"""Round 3 (research/preregistration.md, "Runda 3"): families R3A..R3H on BTC and ETH perps, 15m execution bars.

Run: python -m cqs.research.run_round3 [R3A ...]   (R3X: the pre-registered equal-weight combination)
  -> research/results/R3*.json and R3*_oos.csv (stitched out-of-sample daily returns) per family, and
     research/results/R3_summary.json (program-level Deflated Sharpe and the acceptance checklist) once all
     eight families are in the registry. A family already in the registry is never rerun.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cqs.data import store
from cqs.research import metrics_window, registry, stats
from cqs.research.run_ext import MAJORS, _fp, _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.strategies import intraday as I

COLS = ["open", "high", "low", "close", "volume"]
H1, H4 = pd.Timedelta(hours=1), pd.Timedelta(hours=4)
DEV_END = "2025-05-31"


@dataclass
class Data:
    m15: dict
    h1: dict
    h4: dict
    dclose: dict
    index15: pd.DatetimeIndex
    bars: tuple
    funding: pd.DataFrame


def load_data() -> Data:
    m15, h1, h4, dclose = {}, {}, {}, {}
    for s in MAJORS:
        m15[s] = store.load_klines(s, "15m").set_index("open_time")[COLS]
        h1[s] = store.load_klines(s, "1h").set_index("open_time")[COLS]
        h4[s] = I.resample(h1[s], "4h")
        dclose[s] = store.load_klines(s, "1d").set_index("open_time")["close"]
    idx = m15[MAJORS[0]].index
    for s in MAJORS[1:]:
        idx = idx.union(m15[s].index)
    bars = tuple(pd.DataFrame({s: m15[s][f] for s in MAJORS}).reindex(idx) for f in ("open", "high", "low", "close"))
    funding = pd.DataFrame({s: store.load_funding(s).set_index("funding_time")["rate"] for s in MAJORS}).sort_index()
    return Data(m15, h1, h4, dclose, idx, bars, funding)


# --------------------------------------------------------------------------- variant builders: (data, symbol)
def _r3a(d, s, M, target, killzones):
    return I.r3a_pdh_sweep(d.m15[s], M, target, killzones)


def _r3b(d, s, M, target):
    return I.r3b_asia_sweep(d.m15[s], M, target)


def _r3c(d, s, interval, trend_filter):
    df = d.m15[s] if interval == "15m" else d.h1[s]
    tr = I.daily_trend(d.dclose[s], df.index, 50) if trend_filter else None
    w, st = I.r3c_fvg(df, tr)
    return (w, st) if interval == "15m" else I.to_15m(w, st, df.index, H1, d.m15[s].index)


def _r3d(d, s, k, interval):
    df, bar = (d.h1[s], H1) if interval == "1h" else (d.h4[s], H4)
    return I.to_15m(*I.r3d_structure(df, k), df.index, bar, d.m15[s].index)


def _r3e(d, s, session, minutes):
    return I.r3e_orb(d.m15[s], session, minutes)


def _r3f(d, s, hour):
    return I.r3f_intraday_momentum(d.m15[s], d.h1[s], hour)


def _r3g(d, s, K):
    return I.r3g_vwap(d.m15[s], d.h1[s], K)


def _r3h(d, s, T, S):
    df = d.h4[s]
    return I.to_15m(*I.r3h_pullback(df, I.daily_trend(d.dclose[s], df.index, S), T), df.index, H4, d.m15[s].index)


GRIDS = {
    "R3A": ("r3a_pdh_sweep", _r3a, [dict(M=M, target=t, killzones=k) for M in (1, 4) for t in ("2R", "mid")
                                    for k in (False, True)]),
    "R3B": ("r3b_asia_sweep", _r3b, [dict(M=M, target=t) for M in (1, 2) for t in ("opposite", "2R")]),
    "R3C": ("r3c_fvg", _r3c, [dict(interval=iv, trend_filter=f) for iv in ("15m", "1h") for f in (False, True)]),
    "R3D": ("r3d_structure", _r3d, [dict(k=k, interval=iv) for k in (3, 5) for iv in ("1h", "4h")]),
    "R3E": ("r3e_orb", _r3e, [dict(session=s, minutes=m) for s in ("utc", "ny") for m in (15, 30, 60)]),
    "R3F": ("r3f_intraday_momentum", _r3f, [dict(hour=h) for h in (12, 16, 20)]),
    "R3G": ("r3g_vwap", _r3g, [dict(K=k) for k in (2.0, 3.0)]),
    "R3H": ("r3h_pullback", _r3h, [dict(T=t, S=s) for t in (5, 10) for s in (50, 200)]),
}
assert sum(len(g[2]) for g in GRIDS.values()) == 35  # the pre-registered number of trials


def build_runs(fam: str, d: Data) -> tuple[list, dict]:
    fname, fn, grid = GRIDS[fam]
    runs, entries = [], {}
    for params in grid:
        W, S = {}, {}
        for s in MAJORS:
            w, st = fn(d, s, **params)
            W[s] = pd.Series(np.asarray(w, float), index=d.m15[s].index).reindex(d.index15)
            S[s] = pd.Series(np.asarray(st, float), index=d.m15[s].index).reindex(d.index15)
        W, S = pd.DataFrame(W), pd.DataFrame(S)
        name = _name(fname, params)
        entries[name] = int(((W != 0) & W.notna()).sum().sum())
        runs.append((name, params, W, S))
    return runs, entries


def fingerprint() -> str:
    return _fp(MAJORS, interval="15m") + _fp(MAJORS, interval="1h") + _fp(MAJORS)


def run_family(fam: str, d: Data, log: bool = True) -> tuple[dict, pd.Series]:
    runs, entries = build_runs(fam, d)
    out = evaluate_runs(fam, runs, d.bars, d.funding, d.dclose["BTCUSDT"], "BTC+ETH 15m", fingerprint(),
                        log=log, keep_series=True, engine_kwargs={"reenter_after_stop": True})
    oos = out.pop("_oos")
    out.pop("_oos_x2", None)
    for name, n in entries.items():  # every decision of the strategy must reach the engine
        v = out["variants"][name]
        v["entries"] = n
        if "n_trades" in v and v["n_trades"] != n:
            v["engine_trade_mismatch"] = v["n_trades"] - n
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


# ------------------------------------------------------------------------------------- program-level verdict
def _ok(x, f) -> bool:
    return x is not None and np.isfinite(x) and bool(f(x))


def checklist(out: dict, dsr_r3: float) -> dict:
    oos = out["walk_forward"]["oos"] or {}
    x2 = out["walk_forward"]["oos_costs_x2"] or {}
    reg = out.get("regimes_oos") or {}
    bench = out.get("benchmarks_same_window") or {}
    best_bench = max((v["sharpe"] for v in bench.values() if v.get("sharpe") is not None), default=np.nan)
    sr = oos.get("sharpe")
    return {
        "oos_sharpe_ge_1": _ok(sr, lambda x: x >= 1.0),
        "dsr_ge_0_95": _ok(dsr_r3, lambda x: x >= 0.95),
        "pbo_le_0_25": _ok(out.get("pbo"), lambda x: x <= 0.25),
        "max_dd_within_25pct": _ok(oos.get("max_drawdown"), lambda x: x >= -0.25),
        "positive_at_costs_x2": _ok(x2.get("total_return"), lambda x: x > 0),
        "sharpe_ge_0_bull_bear_range": all(_ok((reg.get(k) or {}).get("sharpe"), lambda x: x >= 0)
                                           for k in ("bull", "bear", "range")),
        "beats_benchmarks": _ok(sr, lambda x: np.isfinite(best_bench) and x > best_bench),
    }


def summarize(results: dict, oos: dict) -> dict:
    ex = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    n_all = len(ex)
    sr_all = [r["sharpe_per_period"] for r in ex if r.get("sharpe_per_period") is not None]
    sr_r3 = [r["sharpe_per_period"] for r in ex
             if r.get("family", "").startswith("R3") and r.get("sharpe_per_period") is not None]
    var_r3, var_all = float(np.var(sr_r3, ddof=1)), float(np.var(sr_all, ddof=1))
    fams = {}
    for fam, out in results.items():
        r = oos[fam].to_numpy()
        dsr_r3 = stats.deflated_sharpe(r, n_all, var_r3) if len(r) > 30 else float("nan")
        dsr_all = stats.deflated_sharpe(r, n_all, var_all) if len(r) > 30 else float("nan")
        checks = checklist(out, dsr_r3)
        fams[fam] = {"oos": out["walk_forward"]["oos"], "oos_costs_x2": out["walk_forward"]["oos_costs_x2"],
                     "pbo": out["pbo"], "dsr_program_round3_variance": dsr_r3,
                     "dsr_program_full_registry_variance": dsr_all, "regimes_oos": out["regimes_oos"],
                     "benchmarks_same_window": out.get("benchmarks_same_window"),
                     "checks": checks, "candidate": all(checks.values())}
    return {"n_trials_program": n_all, "n_trials_round3": len(sr_r3), "sharpe_var_round3": var_r3,
            "sharpe_var_registry": var_all,
            "expected_max_sharpe_annual_round3_var": stats.expected_max_sharpe(n_all, var_r3) * np.sqrt(365),
            "expected_max_sharpe_annual_registry_var": stats.expected_max_sharpe(n_all, var_all) * np.sqrt(365),
            "families": fams, "candidates": [f for f, v in fams.items() if v["candidate"]]}


# ------------------------------------------------------------------------------ R3X: equal-weight combination
def run_r3x(d: Data | None = None, log: bool = True) -> dict:
    """Pre-registered R3X: equal-weight mean of the eight families' stitched out-of-sample daily returns (each
    family with the variants its own walk-forward chose). One trial. The 2x-cost series re-simulates the chosen
    variants at doubled costs (the same trials, not logged again)."""
    from cqs.backtest import engine, metrics
    from cqs.research import walkforward

    fams = list(GRIDS)
    results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in fams}
    oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in fams}
    d = d or load_data()
    oos2 = {}
    for f in fams:
        runs = {r[0]: r for r in build_runs(f, d)[0]}
        cache, parts = {}, []
        for ch in results[f]["walk_forward"]["choices"]:
            v = ch["variant"]
            if v not in cache:
                _, _, W, S = runs[v]
                res = engine.run(*d.bars, W, COSTS.scaled(2), stops=S, funding=d.funding, reenter_after_stop=True)
                cache[v] = metrics.daily_returns(res.equity)
            r = cache[v]
            t0, t1 = (pd.Timestamp(ch[k]) for k in ("test_start", "test_end"))
            t0, t1 = (t if t.tzinfo else t.tz_localize("UTC") for t in (t0, t1))
            parts.append(r[(r.index >= t0) & (r.index < t1)])
        oos2[f] = pd.concat(parts)
    X = pd.DataFrame(oos).sort_index().fillna(0.0).mean(axis=1)
    X2 = pd.DataFrame(oos2).reindex(X.index).fillna(0.0).mean(axis=1)
    reg = walkforward.regimes(d.dclose["BTCUSDT"])
    out = {"family": "R3X", "components": fams,
           "walk_forward": {"oos": walkforward.summary(X), "oos_costs_x2": walkforward.summary(X2)},
           "component_pbo": {f: results[f]["pbo"] for f in fams},
           "regimes_oos": walkforward.by_regime(X, reg),
           "benchmarks_same_window": metrics_window.benchmarks_in_window(str(X.index[0].date()), DEV_END, COSTS)}
    if log:
        registry.log_experiment("R3X", "r3x_equal_weight(R3A..R3H)", {"families": fams, "market": "BTC+ETH 15m"},
                                out["walk_forward"]["oos"], fingerprint(), (str(X.index[0].date()), str(X.index[-1].date())),
                                notes="out-of-sample series only (walk-forward of each component)",
                                sharpe_per_period=stats.sharpe_per_period(X.to_numpy()))
    ex = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    sr_r3 = [r["sharpe_per_period"] for r in ex if r.get("family", "") in GRIDS and r.get("sharpe_per_period") is not None]
    dsr = stats.deflated_sharpe(X.to_numpy(), len(ex), float(np.var(sr_r3, ddof=1)))
    checks = checklist(out | {"pbo": 0.0}, dsr)  # PBO is undefined for a single strategy: reported per component
    checks.pop("pbo_le_0_25")
    out.update({"dsr_program_round3_variance": dsr, "n_trials_program": len(ex), "checks": checks,
                "candidate": all(checks.values())})
    X.rename("R3X").to_csv(RESULTS_DIR / "R3X_oos.csv")
    (RESULTS_DIR / "R3X.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
    return out


def _done(fam: str) -> bool:
    """A family whose trials are already in the registry is never run (or logged) again."""
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    fams = (argv if argv is not None else sys.argv[1:]) or list(GRIDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d, ran = None, False
    if "R3X" in fams:  # the pre-registered combination, once all eight families are in the registry
        fams = [f for f in fams if f != "R3X"]
        if _done("R3X"):
            print("R3X already in the registry: kept, not rerun", flush=True)
        else:
            x = run_r3x()
            print("R3X", x["walk_forward"]["oos"], "candidate:", x["candidate"], flush=True)
    for fam in fams:
        if _done(fam):
            print(fam, "already in the registry: kept, not rerun", flush=True)
            continue
        d = d or load_data()
        t = time.time()
        out, o = run_family(fam, d)
        o.rename(fam).to_csv(RESULTS_DIR / f"{fam}_oos.csv")
        path = RESULTS_DIR / f"{fam}.json"
        path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(path, f"{time.time() - t:.0f}s", flush=True)
        ran = True
    if (ran or not (RESULTS_DIR / "R3_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = summarize(results, oos)
        (RESULTS_DIR / "R3_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round3_summary", {"candidates": summ["candidates"],
                                              "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
