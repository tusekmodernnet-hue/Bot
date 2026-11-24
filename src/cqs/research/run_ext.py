"""Runner for H3b, H4, H5 and H7 (see research/preregistration.md, section Zmiany)."""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from cqs import config
from cqs.backtest.costs import CostModel
from cqs.data import external, store
from cqs.research import panel as panel_mod
from cqs.research import registry
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.strategies import families, families_ext

MAJORS = ["BTCUSDT", "ETHUSDT"]


def majors_panel():
    p = panel_mod.load(MAJORS)
    p.universe = p.close.notna()
    return p


def _fp(symbols, market="futures/um", interval="1d"):
    return store.data_fingerprint([store.parquet_path(market, store.KLINES, s, interval) for s in symbols])


def _name(fname, params):
    return f"{fname}({', '.join(f'{k}={v}' for k, v in params.items())})"


def run_h3b():
    p = majors_panel()
    runs = []
    for z in (1.5, 2.0):
        for hd in (3, 7):
            params = dict(z=z, hold_days=hd)
            runs.append((_name("h3b_crowding_ts", params), params, families_ext.h3b_crowding_ts(p, **params), None))
    return evaluate_runs("H3b", runs, (p.open, p.high, p.low, p.close), p.funding, p.close["BTCUSDT"],
                         "BTC+ETH", _fp(MAJORS), rebalance_band=0.02)


def run_h4():
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
    runs = []
    for t in (0.05, 0.10, 0.20):
        w, _ = families_ext.h4_carry(p, sc, t)
        runs.append((_name("h4_carry", dict(threshold=t)), dict(threshold=t), w, None))
    fp = _fp(MAJORS) + _fp(MAJORS, market="spot")
    return evaluate_runs("H4", runs, tuple(bars), p.funding, p.close["BTCUSDT"], "BTC+ETH spot/perp", fp,
                         rebalance_band=0.02, costs=costs)


def run_h5():
    out = {}
    syms = open(config.DATA_DIR / "universe_ever_top50.txt").read().split()
    for base, panel_fn, label in (("H1", majors_panel, "BTC+ETH"), ("H2a", lambda: panel_mod.load(syms), "universe"),
                                  ("H3a", lambda: panel_mod.load(syms), "universe")):
        p = panel_fn()
        runs = []
        for fname, params in families.GRIDS[base]:
            prm = params | {"vol_overlay": False}
            runs.append((_name(fname, prm), prm, families.FUNCS[fname](p, **prm), None))
        out[base] = evaluate_runs("H5", runs, (p.open, p.high, p.low, p.close), p.funding, p.close["BTCUSDT"],
                                  f"{label}, no vol overlay ({base})", _fp(list(p.close.columns)), rebalance_band=0.02)
    return out


def run_h7():
    p = majors_panel()
    fg, mvrv, vix = external.fear_greed(), external.coinmetrics(), external.fred("VIXCLS")
    gates = {
        "fear_greed_extreme": (fg < 20) | (fg > 80),
        "mvrv_above_3": mvrv > 3.0,
        "vix_above_30": (vix > 30).reindex(pd.date_range(vix.index[0], vix.index[-1], freq="1D")).ffill().astype(bool),
    }
    runs = []
    for gname, block in gates.items():
        for fname, params in families.GRIDS["H1"]:
            w = families_ext.gate(families.FUNCS[fname](p, **params), block)
            prm = params | {"gate": gname}
            runs.append((_name(fname, prm), prm, w, None))
    return evaluate_runs("H7", runs, (p.open, p.high, p.low, p.close), p.funding, p.close["BTCUSDT"],
                         "BTC+ETH gated H1", _fp(MAJORS) + "+ext", rebalance_band=0.02)


def main():
    which = sys.argv[1:] or ["H3b", "H4", "H5", "H7"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for fam in which:
        t = time.time()
        out = {"H3b": run_h3b, "H4": run_h4, "H5": run_h5, "H7": run_h7}[fam]()
        path = RESULTS_DIR / f"{fam}.json"
        path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(path, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
