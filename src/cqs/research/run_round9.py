"""Round 9 (research/preregistration.md, "Runda 9"): risk control for the shorts of new perpetuals on daily bars.

R9A  the R7C and R8B events in one pool, shorted at the close of D0 + 1, -0.05 of equity each, at most 20 at a time,
     30 days in position, hedged with the top-20 alt basket of round 8, and a close-based stop at (1 + k) x the entry
     open, k in {none, 0.5, 1.0}.
Measured in excess of T-bills on the margin (50% of gross notional, at most the equity), as in rounds 7 and 8. The run
goes through the engine: next-day open, taker fee, slippage, actual funding, and a rebalance band of 20% of each
target notional (1% of equity for a -0.05 short).

Run: python -m cqs.research.run_round9  -> research/results/R9A.json, R9A_oos.csv, R9_summary.json
A family already in the registry is never rerun. The runner refuses to start unless the round-9 section of the
pre-registration is committed, and refuses a family whose traded contracts lack funding data.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from cqs.research import metrics_window, registry
from cqs.research import run_round7 as R7
from cqs.research import run_round8 as R8
from cqs.research.run_ext import _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.research.run_round3 import DEV_END
from cqs.strategies import listing_risk, listings

BAND_REL = 0.20
GRIDS = {"R9A": ("r9a_pooled_listing_shorts_stop", [dict(stop=None), dict(stop=0.5), dict(stop=1.0)])}
LABELS = {"R9A": "new perpetuals (R7C + R8B events): shorts -0.05, max 20, d=1, H=30, close stop + top20 alt basket "
                 "long (BTC excluded), 1d"}


def load_r9a() -> dict:
    first_perp, first_spot = R7.listing_dates()
    ev7 = listings.new_token_events(first_perp, first_spot, R7.LISTINGS_AFTER, R8._until())
    ev8, checked, inconsistent = R8.r8b_events(first_perp, first_spot)
    events = listing_risk.pooled_events(ev7, ev8)
    return R8._event_family(events, {"checked": checked, "inconsistent": inconsistent,
                                     "events_r7c": int(len(ev7)), "events_r8b": int(len(ev8))})


def r9a_runs(d: dict):
    fname, grid = GRIDS["R9A"]
    p = d["p"]
    runs, stats_ = [], {}
    for params in grid:
        W, held, stops = listing_risk.r9a(p.open, p.close, p.quote_volume, d["events"], d["basket"], params["stop"])
        W = W.reindex(columns=p.close.columns, fill_value=0.0)
        name = _name(fname, params)
        runs.append((name, params, W, None))
        n = (W < 0).sum(axis=1)
        on = n > 0
        stats_[name] = {"events": int(len(d["events"])), "traded": int((W < 0).any().sum()),
                        "mean_shorts": float(n.mean()),
                        "days_at_capacity": float((n >= listing_risk.MAX_SHORTS).mean()),
                        "mean_basket_names": float((W > 0).sum(axis=1)[on].mean()) if on.any() else 0.0} | \
            listing_risk.stop_stats(stops, p.open)
    return runs, stats_


LOADERS = {"R9A": (load_r9a, r9a_runs)}


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
    for k in ("checked", "inconsistent", "events_r7c", "events_r8b"):
        if k in d:
            out[k] = d[k]
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


def preregistered() -> bool:
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and "## Runda 9" in path.read_text(encoding="utf-8")


def _done(fam: str) -> bool:
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not preregistered():
        raise SystemExit("round 9 is not pre-registered in a committed research/preregistration.md")
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
    if (ran or not (RESULTS_DIR / "R9_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = R7.summarize(results, oos)
        (RESULTS_DIR / "R9_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round9_summary", {"candidates": summ["candidates"],
                                              "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
