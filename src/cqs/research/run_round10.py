"""Round 10 (research/preregistration.md, "Runda 10"): short liquid tokens after an extreme one-day rise, daily bars.

R10A  events: a close-to-close rise of at least X in one day by a contract of the liquid universe (age >= 90 days,
      30-day average quote volume >= 20M USDT); short at the close of the next day, -0.05 of equity each, at most 20
      at a time, H days in position, the R9A stop (close >= 2 x entry open), hedged with the top-20 alt basket.
      X in {30%, 50%}, H in {10, 30}.
Measured in excess of T-bills on the margin (50% of gross notional, at most the equity), as in rounds 7 to 9, through
the engine: next-day open, taker fee, slippage, actual funding, a rebalance band of 20% of each target notional.

Run: python -m cqs.research.run_round10  -> research/results/R10A.json, R10A_oos.csv, R10_summary.json
A family already in the registry is never rerun. The runner refuses to start unless the round-10 section of the
pre-registration is committed, and refuses a family whose traded contracts lack funding data.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from cqs import config
from cqs.data import store
from cqs.data.universe import UniverseRules, daily_panel, eligible
from cqs.research import metrics_window, registry
from cqs.research import run_round7 as R7
from cqs.research import run_round8 as R8
from cqs.research.run_ext import _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.research.run_round3 import DEV_END
from cqs.strategies import listing_risk, listings, pumps, supply

BAND_REL = 0.20
THRESHOLDS = (0.3, 0.5)
GRIDS = {"R10A": ("r10a_pump_fade", [dict(x=0.3, hold=10), dict(x=0.3, hold=30),
                                     dict(x=0.5, hold=10), dict(x=0.5, hold=30)])}
LABELS = {"R10A": "liquid universe (age >= 90d, 20M): short after a one-day rise >= X, d=1, -0.05, max 20, "
                  "close stop 2x + top20 alt basket long (BTC excluded), 1d"}


def liquid_contracts() -> list[str]:
    perps = sorted(x.name for x in (config.PARQUET_DIR / "futures_um" / store.KLINES).iterdir())
    return [s for s in perps if s not in supply.EXCLUDED and s != listings.HEDGE and "_" not in s]


def load_r10a() -> dict:
    dp = daily_panel(liquid_contracts())
    uni = eligible(dp, UniverseRules(top_n=None))
    uni = uni[[c for c in uni.columns if bool(uni[c].any())]]
    events = {x: pumps.pump_events(dp["close"], uni, x) for x in THRESHOLDS}
    syms = sorted(set().union(*(set(e["contract"]) for e in events.values())))
    return R8._event_family(pd.Series(index=syms, dtype=object),
                            {"pump_events": events, "events_count": {str(x): int(len(e)) for x, e in events.items()}})


def r10a_runs(d: dict):
    fname, grid = GRIDS["R10A"]
    p = d["p"]
    runs, stats_ = [], {}
    for params in grid:
        ev = d["pump_events"][params["x"]]
        W, held, stops = pumps.r10a(p.open, p.close, p.quote_volume, ev, d["basket"], params["hold"])
        W = W.reindex(columns=p.close.columns, fill_value=0.0)
        name = _name(fname, params)
        runs.append((name, params, W, None))
        n = (W < 0).sum(axis=1)
        on = n > 0
        opened = int((held & ~held.shift(1, fill_value=False)).sum().sum())
        stats_[name] = {"events": int(len(ev)), "shorts_opened": opened, "mean_shorts": float(n.mean()),
                        "days_at_capacity": float((n >= pumps.MAX_SHORTS).mean()),
                        "mean_basket_names": float((W > 0).sum(axis=1)[on].mean()) if on.any() else 0.0} | \
            listing_risk.stop_stats(stops, p.open)
    return runs, stats_


LOADERS = {"R10A": (load_r10a, r10a_runs)}


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
    if "events_count" in d:
        out["events_count"] = d["events_count"]
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


def preregistered() -> bool:
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and "## Runda 10" in path.read_text(encoding="utf-8")


def _done(fam: str) -> bool:
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not preregistered():
        raise SystemExit("round 10 is not pre-registered in a committed research/preregistration.md")
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
    if (ran or not (RESULTS_DIR / "R10_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = R7.summarize(results, oos)
        (RESULTS_DIR / "R10_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round10_summary", {"candidates": summ["candidates"],
                                               "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
