"""Round 11 (research/preregistration.md, "Runda 11"): short older liquid tokens after a day of extremely negative
funding, daily bars.

R11A  events: a contract of the liquid universe (age >= 90 days, 30-day average quote volume >= 20M USDT) whose funding
      of the day (sum of the rates settled from 00:00 to 24:00 UTC) is at or below -X; short at the close of the next
      day, -0.05 of equity each, at most 20 at a time, H days in position, the R9A stop (close >= 2 x entry open),
      hedged with the top-20 alt basket. X in {0.3%, 0.6%} a day, H in {10, 30}.
Measured in excess of T-bills on the margin (50% of gross notional, at most the equity), as in rounds 7 to 10, through
the engine: next-day open, taker fee, slippage, actual funding, a rebalance band of 20% of each target notional.

Run: python -m cqs.research.run_round11  -> research/results/R11A.json, R11A_oos.csv, R11_summary.json
A family already in the registry is never rerun. The runner refuses to start unless the round-11 section of the
pre-registration is committed, and refuses to run when a contract of the liquid universe lacks funding data.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from cqs.data.universe import UniverseRules, daily_panel, eligible
from cqs.research import metrics_window, registry
from cqs.research import run_round7 as R7
from cqs.research import run_round8 as R8
from cqs.research.run_ext import _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.research.run_round3 import DEV_END
from cqs.research.run_round10 import liquid_contracts
from cqs.strategies import listing_risk, listings, pumps, short_demand

BAND_REL = 0.20
THRESHOLDS = (0.003, 0.006)
GRIDS = {"R11A": ("r11a_negative_funding_short", [dict(x=0.003, hold=10), dict(x=0.003, hold=30),
                                                  dict(x=0.006, hold=10), dict(x=0.006, hold=30)])}
LABELS = {"R11A": "liquid universe (age >= 90d, 20M): short after a day of funding <= -X, d=1, -0.05, max 20, "
                  "close stop 2x + top20 alt basket long (BTC excluded), 1d"}


def load_r11a() -> dict:
    dp = daily_panel(liquid_contracts())
    uni = eligible(dp, UniverseRules(top_n=None))
    uni = uni[[c for c in uni.columns if bool(uni[c].any())]]
    funding, missing = short_demand.load_funding(list(uni.columns))
    idx = dp["close"].index
    events = {x: short_demand.funding_events(funding, uni, x, idx) for x in THRESHOLDS}
    syms = sorted(set().union(*(set(e["contract"]) for e in events.values())))
    d = R8._event_family(pd.Series(index=syms, dtype=object),
                         {"pump_events": events, "events_count": {str(x): int(len(e)) for x, e in events.items()},
                          "universe_contracts": int(uni.shape[1])})
    d["missing_funding"] = sorted(set(d["missing_funding"]) | set(missing))
    return d


def r11a_runs(d: dict):
    fname, grid = GRIDS["R11A"]
    p = d["p"]
    daily_f = short_demand.daily_funding(p.funding, p.close.index) if len(p.funding.columns) else None
    runs, stats_ = [], {}
    for params in grid:
        ev = d["pump_events"][params["x"]]
        W, held, stops = short_demand.r11a(p.open, p.close, p.quote_volume, ev, d["basket"], params["hold"])
        W = W.reindex(columns=p.close.columns, fill_value=0.0)
        name = _name(fname, params)
        runs.append((name, params, W, None))
        n = (W < 0).sum(axis=1)
        on = n > 0
        opened = int((held & ~held.shift(1, fill_value=False)).sum().sum())
        st = {"events": int(len(ev)), "shorts_opened": opened, "mean_shorts": float(n.mean()),
              "days_at_capacity": float((n >= pumps.MAX_SHORTS).mean()),
              "mean_basket_names": float((W > 0).sum(axis=1)[on].mean()) if on.any() else 0.0}
        if daily_f is not None:
            # funding rate of the day on each held short (held at the close of t -> in position on day t + 1);
            # negative means the shorts pay (diagnostic only)
            h = held.shift(1, fill_value=False).reindex(index=daily_f.index, columns=daily_f.columns,
                                                        fill_value=False).to_numpy(dtype=bool)
            vals = daily_f.to_numpy(dtype=float)[h]
            vals = vals[np.isfinite(vals)]
            st["mean_daily_funding_of_held_shorts"] = float(vals.mean()) if len(vals) else 0.0
        stats_[name] = st | listing_risk.stop_stats(stops, p.open)
    return runs, stats_


LOADERS = {"R11A": (load_r11a, r11a_runs)}


def run_family(fam: str, d: dict, log: bool = True):
    runs, decision_stats = LOADERS[fam][1](d)
    p = d["p"]
    out = evaluate_runs(fam, runs, d["bars"], p.funding, p.close[listings.HEDGE], LABELS[fam], d["fingerprint"],
                        log=log, rebalance_band=0.0, costs=d["costs"],
                        return_adjust=lambda res: R7.tbill_on_margin(res, d["rf"]), keep_series=True,
                        engine_kwargs={"rebalance_band_rel": BAND_REL})
    oos = out.pop("_oos")
    out.pop("_oos_x2", None)
    out["decisions"] = decision_stats
    out["measure"] = "excess over T-bills on margin"
    out["rebalance"] = f"band {BAND_REL:.0%} of each target notional; entries and exits always trade"
    for k in ("events_count", "universe_contracts"):
        if k in d:
            out[k] = d[k]
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


def preregistered() -> bool:
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and "## Runda 11" in path.read_text(encoding="utf-8")


def _done(fam: str) -> bool:
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not preregistered():
        raise SystemExit("round 11 is not pre-registered in a committed research/preregistration.md")
    fams = args or list(GRIDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ran = False
    for fam in fams:
        if _done(fam):
            print(fam, "already in the registry: kept, not rerun", flush=True)
            continue
        t0 = time.time()
        d = LOADERS[fam][0]()
        if d["missing_funding"]:
            raise SystemExit(f"{fam}: no funding data for {len(d['missing_funding'])} contracts, "
                             f"e.g. {d['missing_funding'][:5]}")
        out, o = run_family(fam, d)
        o.rename(fam).to_csv(RESULTS_DIR / f"{fam}_oos.csv")
        (RESULTS_DIR / f"{fam}.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(fam, f"{time.time() - t0:.0f}s", "OOS:", out["walk_forward"]["oos"], "PBO:", out["pbo"], flush=True)
        ran = True
    if (ran or not (RESULTS_DIR / "R11_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = R7.summarize(results, oos)
        (RESULTS_DIR / "R11_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round11_summary", {"candidates": summ["candidates"],
                                               "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
