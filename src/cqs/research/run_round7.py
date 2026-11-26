"""Round 7 (research/preregistration.md, "Runda 7"): three market-neutral families on daily Binance bars.

R7A  spot-perpetual cash and carry on many coins (point-in-time top 50 with a Binance spot pair), excess over
     T-bills on the capital in the spot leg, as in R2A.
R7B  residual mean reversion against BTC (Avellaneda and Lee 2010) on the point-in-time top 20.
R7C  dollar-neutral shorts of new tokens after their perpetual listing.
R7B and R7C are measured in excess of T-bills on the margin (50% of gross notional, at most the whole equity), as
in round 6. Every run goes through the engine: next-day open, taker fee, slippage, actual funding.

Run: python -m cqs.data.spotmap                 (data: spot daily bars and the listings' funding)
     python -m cqs.research.run_round7 [R7A R7B R7C]  -> research/results/R7*.json, R7*_oos.csv, R7_summary.json
A family already in the registry is never rerun. The runner refuses to start unless the round-7 section of the
pre-registration is committed.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as st

from cqs import config
from cqs.backtest.costs import CostModel
from cqs.data import spotmap, store
from cqs.data.universe import UniverseRules, daily_panel, eligible
from cqs.research import metrics_window, registry, stats
from cqs.research import panel as panel_mod
from cqs.research.run_ext import _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_runs
from cqs.research.run_round2 import tbill
from cqs.research.run_round3 import DEV_END, checklist
from cqs.research.run_round5 import T_DAYS, null_sr0
from cqs.strategies import carry_multi, listings, statarb

MAJORS = ("BTCUSDT", "ETHUSDT")
MARGIN = 0.5
BAND = 0.02
LISTINGS_AFTER = pd.Timestamp("2020-01-01", tz="UTC")  # first day of the archive: older contracts are not new
GRIDS = {
    "R7A": ("r7a_carry_multi", [dict(margin=0.10, slots=4), dict(margin=0.10, slots=8),
                                dict(margin=0.20, slots=4), dict(margin=0.20, slots=8)]),
    "R7B": ("r7b_residual_reversion", [dict(s_in=1.25), dict(s_in=2.0)]),
    "R7C": ("r7c_short_new_tokens", [dict(delay=1, hold=30), dict(delay=1, hold=90),
                                     dict(delay=7, hold=30), dict(delay=7, hold=90)]),
}
LABELS = {"R7A": "top50 with a Binance spot pair, spot long / perp short, 1d",
          "R7B": "top20 residual vs BTC, 1d",
          "R7C": "new-token perpetual shorts + BTC long, 1d"}


def _top50() -> list[str]:
    return (config.DATA_DIR / "universe_ever_top50.txt").read_text().split()


def rf_daily(index: pd.DatetimeIndex) -> pd.Series:
    rf = tbill()
    return rf.reindex(index.union(rf.index)).ffill().reindex(index).fillna(0.0)


def tbill_on_margin(res, rf: pd.Series) -> pd.Series:
    """T-bill yield forgone on the margin (MARGIN x gross notional, at most the equity) held during each day."""
    gross = res.positions.abs().sum(axis=1)
    tied = (MARGIN * gross / res.equity).clip(upper=1.0).resample("1D").mean()
    return tied * rf.reindex(tied.index).ffill().fillna(0.0) / 365.0


def tbill_on_spot(spot_cols: list[str], rf: pd.Series):
    """T-bill yield forgone on the capital in the spot leg held during each day (R2A's measure)."""
    def adjust(res) -> pd.Series:
        tied = (res.positions[spot_cols].abs().sum(axis=1) / res.equity).clip(upper=1.0).resample("1D").mean()
        return tied * rf.reindex(tied.index).ffill().fillna(0.0) / 365.0
    return adjust


def _fp(paths) -> str:
    return store.data_fingerprint([p for p in paths if p.exists()])


# ---------------------------------------------------------------- R7A
def load_r7a():
    syms = _top50()
    p = panel_mod.load(syms)
    dp = daily_panel(syms)
    top = eligible(dp, UniverseRules()).reindex(index=p.close.index, columns=p.close.columns, fill_value=False)
    base = eligible(dp, UniverseRules(top_n=None)).reindex(index=p.close.index, columns=p.close.columns,
                                                           fill_value=False)
    spot_set = spotmap.spot_symbols()
    mult, frames, spot_paths = {}, {k: {} for k in ("open", "high", "low", "close")}, []
    for s in p.close.columns:
        m = spotmap.match(s, spot_set)
        if m is None:
            continue
        try:
            k = store.load_klines(m[0], "1d", market="spot")
        except FileNotFoundError:
            continue
        if k.empty:
            continue
        k = k.set_index("open_time")
        mult[s] = m[1]
        spot_paths.append(store.parquet_path("spot", store.KLINES, m[0], "1d"))
        for f in frames:
            frames[f][s] = k[f]
    spot = {f: pd.DataFrame(v).reindex(p.close.index) for f, v in frames.items()}
    cols = [s for s in p.close.columns if s in spot["close"].columns and s in p.funding.columns]
    avail = carry_multi.pair_available(p.close[cols], spot["close"][cols], mult)
    rf = rf_daily(p.close.index)
    excess = carry_multi.funding_known(p.funding[cols], p.close.index).sub(rf, axis=0)
    bars = tuple(pd.concat([getattr(p, f), spot[f][cols].add_suffix(carry_multi.SPOT)], axis=1)
                 for f in ("open", "high", "low", "close"))
    spot_cols = [c + carry_multi.SPOT for c in cols]
    slip = {"BTCUSDT": 2.0, "ETHUSDT": 2.0} | {c: 15.0 for c in spot_cols} | {"BTCUSDT_SPOT": 7.0, "ETHUSDT_SPOT": 7.0}
    costs = CostModel(default_slippage_bps=10.0, slippage_bps=slip)
    perp_paths = [store.parquet_path("futures/um", store.KLINES, s, "1d") for s in syms] + \
                 [store.parquet_path("futures/um", store.FUNDING, s) for s in syms]
    return dict(p=p, bars=bars, cols=cols, excess=excess, can_open=top[cols] & avail, can_hold=base[cols] & avail,
                spot_cols=spot_cols, rf=rf, costs=costs, fingerprint=_fp(perp_paths + spot_paths), mult=mult)


def r7a_runs(d: dict):
    fname, grid = GRIDS["R7A"]
    runs, stats_ = [], {}
    for params in grid:
        W, held = carry_multi.r7a(d["excess"], d["can_open"], d["can_hold"], params["margin"], params["slots"])
        W = W.reindex(columns=d["bars"][3].columns, fill_value=0.0)
        name = _name(fname, params)
        runs.append((name, params, W, None))
        n = held.sum(axis=1)
        stats_[name] = {"days_with_a_pair": float((n > 0).mean()), "mean_pairs": float(n.mean()),
                        "coins_used": int(held.any().sum()), "entries": int((held & ~held.shift(1, fill_value=False)).sum().sum())}
    return runs, stats_


# ---------------------------------------------------------------- R7B
def load_r7b():
    syms = _top50()
    p = panel_mod.load(syms)
    dp = daily_panel(syms)
    uni = eligible(dp, UniverseRules(top_n=20)).reindex(index=p.close.index, columns=p.close.columns, fill_value=False)
    coins = [c for c in p.close.columns if c != statarb.HEDGE and uni[c].any()]
    logp = np.log(p.close)
    logret = logp[coins].diff()
    btc = logp[statarb.HEDGE].diff()
    paths = [store.parquet_path("futures/um", store.KLINES, s, "1d") for s in syms] + \
            [store.parquet_path("futures/um", store.FUNDING, s) for s in syms]
    return dict(p=p, bars=(p.open, p.high, p.low, p.close), logret=logret, btc=btc, uni=uni[coins],
                rf=rf_daily(p.close.index), costs=CostModel(default_slippage_bps=10.0, slippage_bps={s: 2.0 for s in MAJORS}),
                fingerprint=_fp(paths))


def r7b_runs(d: dict):
    fname, grid = GRIDS["R7B"]
    s, b, v = statarb.scores(d["logret"], d["btc"], d["uni"])
    runs, stats_ = [], {}
    for params in grid:
        sign = statarb.positions(s, v, d["uni"], params["s_in"])
        W = statarb.weights(sign, b).reindex(columns=d["bars"][3].columns, fill_value=0.0)
        name = _name(fname, params)
        runs.append((name, params, W, None))
        n = (sign != 0).sum(axis=1)
        stats_[name] = {"mean_positions": float(n.mean()), "days_with_a_position": float((n > 0).mean()),
                        "share_valid_models": float(v.to_numpy()[d["uni"].to_numpy()].mean())}
    return runs, stats_


# ---------------------------------------------------------------- R7C
def listing_dates():
    spot_set = spotmap.spot_symbols()
    perps = sorted(x.name for x in (config.PARQUET_DIR / "futures_um" / store.KLINES).iterdir())
    first_perp, first_spot = {}, {}
    for s in perps:
        d0 = spotmap.first_traded_day(s, "futures/um")
        if d0 is None:
            continue
        first_perp[s] = d0
        m = spotmap.match(s, spot_set)
        first_spot[s] = spotmap.first_traded_day(m[0], "spot") if m else None
    return first_perp, first_spot


def load_r7c():
    first_perp, first_spot = listing_dates()
    events = listings.new_token_events(first_perp, first_spot, LISTINGS_AFTER, pd.Timestamp(DEV_END, tz="UTC"))
    syms = list(events.index) + [listings.HEDGE]
    p = panel_mod.load(syms)
    missing_funding = [s for s in events.index if s not in p.funding.columns]
    paths = [store.parquet_path("futures/um", store.KLINES, s, "1d") for s in syms] + \
            [store.parquet_path("futures/um", store.FUNDING, s) for s in syms]
    return dict(p=p, bars=(p.open, p.high, p.low, p.close), events=events, rf=rf_daily(p.close.index),
                costs=CostModel(default_slippage_bps=10.0, slippage_bps={s: 2.0 for s in MAJORS}),
                fingerprint=_fp(paths), missing_funding=missing_funding)


def r7c_runs(d: dict):
    fname, grid = GRIDS["R7C"]
    runs, stats_ = [], {}
    p = d["p"]
    for params in grid:
        held = listings.shorts(p.close, p.quote_volume, d["events"], params["delay"], params["hold"])
        W = listings.weights(held).reindex(columns=d["bars"][3].columns, fill_value=0.0)
        name = _name(fname, params)
        runs.append((name, params, W, None))
        n = held.sum(axis=1)
        stats_[name] = {"events": int(len(d["events"])), "traded": int(held.any().sum()),
                        "mean_shorts": float(n.mean()), "days_at_capacity": float((n >= listings.MAX_SHORTS).mean())}
    return runs, stats_


LOADERS = {"R7A": (load_r7a, r7a_runs), "R7B": (load_r7b, r7b_runs), "R7C": (load_r7c, r7c_runs)}


def run_family(fam: str, d: dict, log: bool = True):
    runs, decision_stats = LOADERS[fam][1](d)
    adjust = tbill_on_spot(d["spot_cols"], d["rf"]) if fam == "R7A" else (lambda res: tbill_on_margin(res, d["rf"]))
    p = d["p"]
    out = evaluate_runs(fam, runs, d["bars"], p.funding, p.close[listings.HEDGE], LABELS[fam], d["fingerprint"],
                        log=log, rebalance_band=BAND, costs=d["costs"], return_adjust=adjust, keep_series=True)
    oos = out.pop("_oos")
    out.pop("_oos_x2", None)
    out["decisions"] = decision_stats
    out["measure"] = "excess over T-bills on the spot leg" if fam == "R7A" else "excess over T-bills on margin"
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


def _regime_ok(reg: dict | None) -> dict | None:
    """Pre-registered: a regime without any position (zero variance) counts as Sharpe 0."""
    if reg is None:
        return None
    out = {}
    for k, v in reg.items():
        v = dict(v)
        if (v.get("sharpe") is None or not np.isfinite(v["sharpe"])) and v.get("days", 0) > 0 \
                and v.get("total_return") == 0.0:
            v["sharpe"] = 0.0
            v["no_position"] = True
        out[k] = v
    return out


def summarize(results: dict, oos: dict) -> dict:
    ex = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    n_all = len(ex)
    sr0 = null_sr0(n_all)
    fams = {}
    for fam, out in results.items():
        r = oos[fam].to_numpy()
        dsr = stats.deflated_sharpe(r, n_all, 1.0 / T_DAYS) if len(r) > 30 else float("nan")
        out = dict(out)
        out["regimes_oos"] = _regime_ok(out.get("regimes_oos"))
        checks = checklist(out, dsr)
        fams[fam] = {"oos": out["walk_forward"]["oos"], "oos_costs_x2": out["walk_forward"]["oos_costs_x2"],
                     "pbo": out["pbo"], "dsr_program_null_variance": dsr, "regimes_oos": out["regimes_oos"],
                     "benchmarks_same_window": out.get("benchmarks_same_window"), "checks": checks,
                     "candidate": all(checks.values())}
    n_oos = max((len(o) for o in oos.values()), default=0)
    needed = (sr0 + st.norm.ppf(0.95) / np.sqrt(n_oos - 1)) * np.sqrt(365) if n_oos > 1 else float("nan")
    return {"n_trials_program": n_all, "dsr_method": "null variance 1/T (as in rounds 5-6)", "T_days": T_DAYS,
            "sr0_annual": sr0 * np.sqrt(365), "oos_sharpe_for_dsr_0_95_annual_if_normal": needed,
            "n_oos_days": n_oos, "families": fams, "candidates": [f for f, v in fams.items() if v["candidate"]]}


def preregistered() -> bool:
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and "## Runda 7" in path.read_text(encoding="utf-8")


def _done(fam: str) -> bool:
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not preregistered():
        raise SystemExit("round 7 is not pre-registered in a committed research/preregistration.md")
    fams = args or list(GRIDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ran = False
    for fam in fams:
        if _done(fam):
            print(fam, "already in the registry: kept, not rerun", flush=True)
            continue
        t0 = time.time()
        d = LOADERS[fam][0]()
        if fam == "R7C" and d["missing_funding"]:
            raise SystemExit(f"R7C: no funding data for {len(d['missing_funding'])} events, e.g. {d['missing_funding'][:5]}")
        out, o = run_family(fam, d)
        o.rename(fam).to_csv(RESULTS_DIR / f"{fam}_oos.csv")
        (RESULTS_DIR / f"{fam}.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(fam, f"{time.time() - t0:.0f}s", "OOS:", out["walk_forward"]["oos"], "PBO:", out["pbo"], flush=True)
        ran = True
    if (ran or not (RESULTS_DIR / "R7_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = summarize(results, oos)
        (RESULTS_DIR / "R7_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round7_summary", {"candidates": summ["candidates"],
                                              "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
