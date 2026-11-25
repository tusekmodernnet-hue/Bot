"""Round 2 (research/preregistration.md, section "Runda 2"): R2A, R2B, R2C.

Run: python -m cqs.research.run_round2 [R2A R2B R2C]  -> research/results/R2*.json, trials logged in the registry.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from cqs import config
from cqs.backtest.costs import CostModel
from cqs.data import external, store
from cqs.research import metrics_window, registry
from cqs.research import panel as panel_mod
from cqs.research.run_ext import MAJORS, _fp, _name, majors_panel
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.strategies import families, families_ext

R2C_WF_START = pd.Timestamp("2021-07-01", tz="UTC")


def tbill() -> pd.Series:
    """3-month T-bill yield (decimal), indexed by the day it is usable (1-day delay)."""
    return external.fred("DTB3") / 100.0


def run_r2a() -> dict:
    p = majors_panel()
    spot = {s: store.load_klines(s, "1d", market="spot").set_index("open_time") for s in MAJORS}
    sc = pd.DataFrame({s: spot[s]["close"] for s in MAJORS}).reindex(p.close.index)
    bars = []
    for fld in ("open", "high", "low", "close"):
        perp = getattr(p, fld)
        sp = pd.DataFrame({f"{s}_SPOT": spot[s][fld] for s in MAJORS}).reindex(perp.index)
        bars.append(pd.concat([perp, sp], axis=1))
    costs = CostModel(default_slippage_bps=10.0, slippage_bps={"BTCUSDT": 2.0, "ETHUSDT": 2.0,
                                                               "BTCUSDT_SPOT": 7.0, "ETHUSDT_SPOT": 7.0})
    rf = tbill()
    rf_day = rf.reindex(p.close.index.union(rf.index)).ffill().reindex(p.close.index)
    spot_cols = [f"{s}_SPOT" for s in MAJORS]

    def tbill_on_spot(res) -> pd.Series:
        """T-bill yield earned by the alternative on the capital tied up in the spot leg during each day."""
        eq = res.equity
        frac = res.positions[spot_cols].sum(axis=1).abs() / eq
        held = frac.resample("1D").last().shift(1).fillna(0.0)  # capital tied during day t = position at t-1
        r = rf_day.reindex(held.index).ffill().fillna(0.0)
        return held * r / 365.0

    runs = []
    for m in (0.025, 0.05, 0.10):
        params = dict(margin=m)
        runs.append((_name("r2a_carry_rf", params), params, families_ext.r2a_carry_rf(p, sc, rf, m), None))
    fp = _fp(MAJORS) + _fp(MAJORS, market="spot")
    out = evaluate_runs("R2A", runs, tuple(bars), p.funding, p.close["BTCUSDT"], "BTC+ETH spot/perp, excess over T-bills",
                        fp, rebalance_band=0.02, costs=costs, return_adjust=tbill_on_spot)
    oos0 = pd.Timestamp(out["walk_forward"]["oos"]["start"], tz="UTC")
    out["tbill_mean_oos"] = float(rf_day[rf_day.index >= oos0].mean())
    return out


def run_r2b() -> dict:
    syms = open(config.DATA_DIR / "universe_ever_top50.txt").read().split()
    p = panel_mod.load(syms)
    runs = []
    for W in (7, 14):
        for R in (7, 14):
            params = dict(window_days=W, rebalance_every=R)
            runs.append((_name("h3a_funding_xs", params), params, families.h3a_funding_xs(p, **params), None))
    fp = store.data_fingerprint([store.parquet_path("futures/um", store.KLINES, s, "1d") for s in p.close.columns])
    return evaluate_runs("R2B", runs, (p.open, p.high, p.low, p.close), p.funding, p.close["BTCUSDT"],
                         f"universe_top50({len(syms)})", fp, rebalance_band=0.02)


def run_r2c() -> dict:
    p = majors_panel()
    dvol = pd.DataFrame({"BTCUSDT": external.deribit_dvol("BTC"), "ETHUSDT": external.deribit_dvol("ETH")})
    runs = []
    for z in (0.0, 1.0):
        for ls in (False, True):
            params = dict(z=z, long_short=ls)
            runs.append((_name("r2c_vrp", params), params, families_ext.r2c_vrp(p, dvol, **params), None))
    out = evaluate_runs("R2C", runs, (p.open, p.high, p.low, p.close), p.funding, p.close["BTCUSDT"], "BTC+ETH",
                        _fp(MAJORS), rebalance_band=0.02, wf_start=R2C_WF_START)
    # benchmarks in the same, shorter out-of-sample window
    oos0 = out["walk_forward"]["oos"]["start"]
    out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(oos0, "2025-05-31", COSTS)
    out["dvol_first_usable"] = str(dvol.dropna(how="all").index[0].date())
    return out


RUNNERS = {"R2A": run_r2a, "R2B": run_r2b, "R2C": run_r2c}


def main(argv=None) -> int:
    fams = (argv if argv is not None else sys.argv[1:]) or list(RUNNERS)
    for fam in fams:
        t = time.time()
        out = RUNNERS[fam]()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{fam}.json"
        path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(path, f"{time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
