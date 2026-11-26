"""Round 8 (research/preregistration.md, "Runda 8"): the supply mechanism behind R7C on daily Binance bars.

R8A  R7C's new-token shorts with an equal-weight long basket of the point-in-time top 20 without BTC.
R8B  shorts after a new perpetual on a token that already traded on Binance spot (Miller 1977), same basket.
R8C  token age: young tokens short, tokens over 730 days old long, dollar neutral, weekly membership.
All three are measured in excess of T-bills on the margin (50% of gross notional, at most the equity), as in R7B
and R7C. Every run goes through the engine: next-day open, taker fee, slippage, actual funding, and a rebalance
band of 20% of each target notional (for a -0.1 short that is R7C's 2% of equity).

Run: python -m cqs.research.run_round8 [R8A R8B R8C]  -> research/results/R8*.json, R8*_oos.csv, R8_summary.json
A family already in the registry is never rerun. The runner refuses to start unless the round-8 section of the
pre-registration is committed, and refuses a family whose traded contracts lack funding data.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from cqs.backtest.costs import CostModel
from cqs.data import spotmap, store
from cqs.data.universe import UniverseRules, daily_panel, eligible
from cqs.research import metrics_window, registry
from cqs.research import panel as panel_mod
from cqs.research import run_round7 as R7
from cqs.research.run_ext import _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.research.run_round3 import DEV_END
from cqs.strategies import listings, supply

BAND_REL = 0.20
EVENT_GRID = [dict(delay=1, hold=30), dict(delay=1, hold=90), dict(delay=7, hold=30), dict(delay=7, hold=90)]
GRIDS = {
    "R8A": ("r8a_new_tokens_alt_hedge", EVENT_GRID),
    "R8B": ("r8b_new_perp_old_token", EVENT_GRID),
    "R8C": ("r8c_token_age", [dict(young_max=180), dict(young_max=365)]),
}
LABELS = {"R8A": "new-token perpetual shorts + top20 alt basket long (BTC excluded), 1d",
          "R8B": "new perpetuals on tokens with Binance spot history: shorts + top20 alt basket long, 1d",
          "R8C": "liquid universe: tokens 90..A days old short, over 730 days long, weekly membership, 1d"}


def _costs() -> CostModel:
    return CostModel(default_slippage_bps=10.0, slippage_bps={s: 2.0 for s in R7.MAJORS})


def _until() -> pd.Timestamp:
    return pd.Timestamp(DEV_END, tz="UTC")


def _paths(syms) -> list:
    return [store.parquet_path("futures/um", store.KLINES, s, "1d") for s in syms] + \
           [store.parquet_path("futures/um", store.FUNDING, s) for s in syms]


# ---------------------------------------------------------------- R8A, R8B
def _event_family(events: pd.Series, extra: dict | None = None) -> dict:
    top = R7._top50()
    basket = eligible(daily_panel(top), UniverseRules(top_n=20))
    basket = basket[[c for c in basket.columns if c != listings.HEDGE and c not in listings.INDEX
                     and c not in listings.STABLE and bool(basket[c].any())]]
    syms = list(dict.fromkeys(list(events.index) + list(basket.columns) + [listings.HEDGE]))
    p = panel_mod.load(syms)
    basket = basket.reindex(index=p.close.index, columns=p.close.columns, fill_value=False).fillna(False)
    missing = [s for s in syms if s not in p.funding.columns]
    return dict(p=p, bars=(p.open, p.high, p.low, p.close), events=events, basket=basket.astype(bool),
                rf=R7.rf_daily(p.close.index), costs=_costs(), fingerprint=R7._fp(_paths(syms)),
                missing_funding=missing) | (extra or {})


def load_r8a() -> dict:
    first_perp, first_spot = R7.listing_dates()
    events = listings.new_token_events(first_perp, first_spot, R7.LISTINGS_AFTER, _until())
    return _event_family(events)


def load_r8b() -> dict:
    first_perp, first_spot = R7.listing_dates()
    spot_set = spotmap.spot_symbols()
    consistent, checked = {}, 0
    for p, d0 in first_perp.items():
        sp = first_spot.get(p)
        if d0 is None or sp is None or not (R7.LISTINGS_AFTER < d0 <= _until()) or not sp < d0 - listings.SPOT_LEAD:
            continue
        m = spotmap.match(p, spot_set)
        if m is None:
            continue
        perp = store.load_klines(p, "1d", market="futures/um").set_index("open_time")["close"]
        spot = store.load_klines(m[0], "1d", market="spot").set_index("open_time")["close"]
        consistent[p] = supply.listing_consistent(perp, spot, m[1], d0)
        checked += 1
    events = supply.old_token_events(first_perp, first_spot, consistent, R7.LISTINGS_AFTER, _until())
    return _event_family(events, {"checked": checked, "inconsistent": sorted(k for k, v in consistent.items() if not v)})


def event_runs(fam: str):
    def runs_fn(d: dict):
        fname, grid = GRIDS[fam]
        p = d["p"]
        runs, stats_ = [], {}
        for params in grid:
            W, _ = supply.r8_event_shorts(p.close, p.quote_volume, d["events"], d["basket"], params["delay"],
                                          params["hold"])
            W = W.reindex(columns=p.close.columns, fill_value=0.0)
            name = _name(fname, params)
            runs.append((name, params, W, None))
            n = (W < 0).sum(axis=1)
            on = n > 0
            stats_[name] = {"events": int(len(d["events"])), "traded": int((W < 0).any().sum()),
                            "mean_shorts": float(n.mean()),
                            "days_at_capacity": float((n >= listings.MAX_SHORTS).mean()),
                            "mean_basket_names": float((W > 0).sum(axis=1)[on].mean()) if on.any() else 0.0}
        return runs, stats_
    return runs_fn


# ---------------------------------------------------------------- R8C
def load_r8c() -> dict:
    first_perp, first_spot = R7.listing_dates()
    birth = supply.token_birth(first_perp, first_spot)
    syms = [s for s in sorted(first_perp) if s not in supply.EXCLUDED]
    uni = eligible(daily_panel(syms), UniverseRules(top_n=None))
    members = [c for c in uni.columns if c != listings.HEDGE and bool(uni[c].any())]
    load = members + [listings.HEDGE]
    p = panel_mod.load(load)
    uni = uni.reindex(index=p.close.index, columns=p.close.columns, fill_value=False).fillna(False).astype(bool)
    uni[listings.HEDGE] = False
    age = supply.age_days(p.close.index, {s: birth.get(s) for s in members})
    missing = [s for s in members if s not in p.funding.columns]
    return dict(p=p, bars=(p.open, p.high, p.low, p.close), universe=uni, age=age, rf=R7.rf_daily(p.close.index),
                costs=_costs(), fingerprint=R7._fp(_paths(load)), missing_funding=missing)


def r8c_runs(d: dict):
    fname, grid = GRIDS["R8C"]
    p = d["p"]
    runs, stats_ = [], {}
    for params in grid:
        W, _ = supply.r8c(d["universe"], d["age"], p.close, params["young_max"])
        W = W.reindex(columns=p.close.columns, fill_value=0.0)
        name = _name(fname, params)
        runs.append((name, params, W, None))
        on = (W != 0).any(axis=1)
        stats_[name] = {"days_with_positions": float(on.mean()),
                        "mean_young": float((W < 0).sum(axis=1)[on].mean()) if on.any() else 0.0,
                        "mean_old": float((W > 0).sum(axis=1)[on].mean()) if on.any() else 0.0,
                        "mean_gross": float(W.abs().sum(axis=1)[on].mean()) if on.any() else 0.0}
    return runs, stats_


LOADERS = {"R8A": (load_r8a, event_runs("R8A")), "R8B": (load_r8b, event_runs("R8B")), "R8C": (load_r8c, r8c_runs)}


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
    for k in ("checked", "inconsistent"):
        if k in d:
            out[k] = d[k]
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


def preregistered() -> bool:
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and "## Runda 8" in path.read_text(encoding="utf-8")


def _done(fam: str) -> bool:
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not preregistered():
        raise SystemExit("round 8 is not pre-registered in a committed research/preregistration.md")
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
    if (ran or not (RESULTS_DIR / "R8_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = R7.summarize(results, oos)
        (RESULTS_DIR / "R8_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round8_summary", {"candidates": summ["candidates"],
                                              "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
