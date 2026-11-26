"""Final test of R9A on the holdout (research/preregistration.md, "Test końcowy R9A").

R9A is not a candidate (DSR 0.59, PBO 0.51), so this test is an exception that only the owner can approve. It runs
only with all of: the pre-registration section committed, a clean code tree, CQS_ALLOW_HOLDOUT=1, the owner's
approval quoted on the command line, and no earlier final test of R9A in the registry (it never reruns).

Strategy: R9A with the stop at k = 1.0, one continuous simulation from 01.2020 to the end of the stored data, scored
only on days from 2025-06-01. Pass: holdout excess Sharpe >= 1.0, max drawdown no worse than -25%, Sharpe at
costs x2 > 0.

Run: CQS_ALLOW_HOLDOUT=1 python -m cqs.research.final_r9a --approval "<the owner's words and date>"
-> research/results/FINAL_R9A.json, FINAL_R9A_holdout.csv, and a "final_test_r9a" event in the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from cqs import config
from cqs.backtest import engine
from cqs.data import guard, store
from cqs.research import metrics_window, registry, stats, walkforward
from cqs.research import run_round7 as R7
from cqs.research import run_round8 as R8
from cqs.research import run_round9 as R9
from cqs.research.run_family import COSTS, RESULTS_DIR, _daily_net
from cqs.strategies import listing_risk, listings

STOP_K = 1.0
SHARPE_MIN = 1.0
MAX_DD = -0.25
HEADING = "## Test końcowy R9A"
EVENT = "final_test_r9a"
VARIANT = "r9a_pooled_listing_shorts_stop(stop=1.0)"


def all_perps() -> list[str]:
    return sorted(x.name for x in (config.PARQUET_DIR / "futures_um" / store.KLINES).iterdir())


def load_full() -> dict:
    """R9A's inputs over every stored day the guard allows (the holdout too, inside a declared final test)."""
    until = pd.Timestamp(guard.resolve_end(None)).normalize()
    first_perp, first_spot = R7.listing_dates()
    ev7 = listings.new_token_events(first_perp, first_spot, R7.LISTINGS_AFTER, until)
    ev8, checked, inconsistent = R8.r8b_events(first_perp, first_spot, until)
    events = listing_risk.pooled_events(ev7, ev8)
    return R8._event_family(events, {"checked": checked, "inconsistent": inconsistent,
                                     "events_r7c": set(ev7.index), "events_r8b": set(ev8.index)},
                            universe=all_perps())


def simulate(d: dict, cost_scale: float = 1.0):
    p = d["p"]
    W, held, stops = listing_risk.r9a(p.open, p.close, p.quote_volume, d["events"], d["basket"], STOP_K)
    W = W.reindex(columns=p.close.columns, fill_value=0.0)
    costs = d["costs"] if cost_scale == 1.0 else d["costs"].scaled(cost_scale)
    res = engine.run(p.open, p.high, p.low, p.close, W, costs, funding=p.funding,
                     rebalance_band_rel=R9.BAND_REL)
    r = _daily_net(res)
    r = r - R7.tbill_on_margin(res, d["rf"]).reindex(r.index).fillna(0.0)
    return r, res, held, stops


def verdict(r: pd.Series, r_x2: pd.Series) -> dict:
    """The pre-registered pass rule on the holdout slice of the daily excess returns."""
    s = walkforward.summary(r)
    s2 = walkforward.ann_sharpe(r_x2)
    checks = {"sharpe_ge_1": bool(s["sharpe"] >= SHARPE_MIN),
              "max_dd_within_25pct": bool(s["max_drawdown"] >= MAX_DD),
              "positive_at_costs_x2": bool(s2 > 0)}
    return {"holdout": s, "sharpe_costs_x2": s2, "checks": checks, "passed": all(checks.values())}


def halves(r: pd.Series) -> dict:
    out = {}
    for (y, h), g in r.groupby([r.index.year, (r.index.month > 6).astype(int) + 1]):
        out[f"{y}H{h}"] = {"days": int(len(g)), "return": float((1 + g).prod() - 1),
                           "sharpe": walkforward.ann_sharpe(g)}
    return out


def report(d: dict, r: pd.Series, res, held: pd.DataFrame, stops: list, start: pd.Timestamp) -> dict:
    """Diagnostics only: they do not enter the verdict."""
    h = r[r.index >= start]
    lr = np.log1p(h)
    top10 = float(lr.sort_values(ascending=False).head(10).sum() / lr.sum()) if lr.sum() > 0 else None
    p = d["p"]
    ret = p.close.pct_change(fill_method=None)
    contrib = (res.positions.shift(1) * ret).fillna(0.0).div(res.equity.shift(1), axis=0)
    contrib = contrib[contrib.index >= start]
    shorted = held.loc[held.index >= start]
    sc = [c for c in shorted.columns if shorted[c].any()]
    in_window = [s for s in stops if s["stopped"] >= start]
    ev = d["events"]
    return {"psr_vs_0": stats.probabilistic_sharpe(h.to_numpy(), 0.0) if len(h) > 30 else None,
            "halves": halves(h), "top10_days_share_of_log_return": top10,
            "events_with_d0_in_holdout": int(sum(pd.Timestamp(v) >= start for v in ev.values)),
            "contracts_shorted_in_holdout": len(sc), "stops_in_holdout": len(in_window),
            "stop_stats": listing_risk.stop_stats(in_window, p.open),
            "short_leg_r7c": float(contrib[[c for c in sc if c in d["events_r7c"]]].sum().sum()),
            "short_leg_r8b": float(contrib[[c for c in sc if c in d["events_r8b"]]].sum().sum())}


def dev_reproduction(r: pd.Series) -> dict:
    """The continuous run up to 2025-05-31 must give the registered full-period Sharpe of the variant (raw returns
    would be needed for an exact match; the excess-return Sharpe is the registered `sharpe_excess`)."""
    reg = [x for x in registry.read() if x.get("type") == "experiment" and x.get("name") == VARIANT]
    want = reg[-1]["metrics"].get("sharpe_excess") if reg else None
    got = walkforward.ann_sharpe(r[r.index < config.HOLDOUT_START])
    return {"registered_sharpe_excess": want, "recomputed": got,
            "match": bool(want is not None and abs(got - want) < 0.02)}


def preregistered() -> bool:
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and HEADING in path.read_text(encoding="utf-8")


def _already_run() -> bool:
    return any(x.get("type") == "event" and x.get("event") == EVENT for x in registry.read())


def main(argv=None, loader=load_full) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--approval", required=True, help="the owner's approval, quoted with its date")
    args = ap.parse_args(argv)
    if len(args.approval.strip()) < 10:
        raise SystemExit("quote the owner's approval (at least 10 characters)")
    if not preregistered():
        raise SystemExit("the final test of R9A is not pre-registered in a committed research/preregistration.md")
    if registry._code_version().get("dirty"):
        raise SystemExit("commit the code first: the registry must record the exact version")
    if _already_run():
        raise SystemExit("the final test of R9A already ran: it is never repeated")
    t0 = time.time()
    with guard.final_test(f"R9A final test (k = {STOP_K}), approved by the owner: {args.approval.strip()}"):
        d = loader()
        if d["missing_funding"]:
            raise SystemExit(f"no funding data for {len(d['missing_funding'])} contracts, e.g. {d['missing_funding'][:5]}")
        r, res, held, stops = simulate(d)
        r_x2, _, _, _ = simulate(d, 2.0)
        start = config.HOLDOUT_START
        out = verdict(r[r.index >= start], r_x2[r_x2.index >= start])
        out["diagnostics"] = report(d, r, res, held, stops, start)
        out["dev_reproduction"] = dev_reproduction(r)
        h = out["holdout"]
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(h["start"], h["end"], COSTS)
        out |= {"strategy": VARIANT, "approval": args.approval.strip(), "fingerprint": d["fingerprint"],
                "measure": "excess over T-bills on margin", "rule": {"sharpe_min": SHARPE_MIN, "max_dd": MAX_DD,
                                                                     "costs_x2_sharpe_gt": 0.0}}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        r[r.index >= start].rename("R9A_final").to_csv(RESULTS_DIR / "FINAL_R9A_holdout.csv")
        (RESULTS_DIR / "FINAL_R9A.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        registry.log_event(EVENT, registry._clean({k: out[k] for k in ("strategy", "holdout", "sharpe_costs_x2",
                                                                       "checks", "passed", "approval")}))
    print("R9A final test", f"{time.time() - t0:.0f}s", "passed:", out["passed"], out["holdout"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
