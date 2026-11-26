"""Round 6 (research/preregistration.md, "Runda 6"): calendar basis of the BTC and ETH quarterlies against the
perpetual's trailing funding, market neutral. R6A is always short basis; R6B trades the spread between the basis
and the trailing funding (theta 2% or 5% a year).

Trials run on Binance USDⓈ-M (perpetual + quarterlies, 1h bars) through the engine: next-bar open, taker fee,
slippage 2 bp on the perpetual and 3 bp on the quarterly, actual funding on the perpetual leg. Returns are measured
in excess of the 3-month T-bill on the capital tied up as margin (50% of the gross notional, at most the whole
equity; the rest is assumed to sit in T-bills). The same rules, with the walk-forward choices made on Binance, run
on Deribit as the pre-registered confirmation; that run is logged as an event, not as a trial.

Run: python -m cqs.research.run_round6 --download-deribit   (data only)
     python -m cqs.research.run_round6 [R6A R6B]  -> research/results/R6*.json, R6*_oos.csv, R6_summary.json
A family already in the registry is never rerun. The runner refuses to start unless the round-6 section of the
pre-registration is committed.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st

from cqs.backtest import engine, metrics
from cqs.backtest.costs import CostModel
from cqs.data import deribit, quarterly, store
from cqs.data.quarterly import Contract
from cqs.research import metrics_window, registry, stats, walkforward
from cqs.research.run_ext import _name
from cqs.research.run_family import COSTS, RESULTS_DIR, evaluate_daily
from cqs.research.run_round3 import DEV_END, checklist
from cqs.research.run_round5 import T_DAYS, null_sr0
from cqs.strategies import basis

COINS = {"BTC": ("BTCUSDT", "BTC-PERPETUAL"), "ETH": ("ETHUSDT", "ETH-PERPETUAL")}
START = pd.Timestamp("2020-01-01", tz="UTC")  # walk-forward origin: the first test window starts on 2022-01-01
BARS_FROM = pd.Timestamp("2021-01-01", tz="UTC")
FUNDING_FROM = pd.Timestamp("2020-11-01", tz="UTC")  # 30-day trailing funding needs history before the first bar
DERIBIT_DELIVERIES = ("2021-03-01", "2025-09-30")
TAKER_FEE = 0.0005
PERP_SLIPPAGE_BPS, QUARTERLY_SLIPPAGE_BPS = 2.0, 3.0
MARGIN = 0.5  # capital tied up per unit of gross notional (2:1 leverage cap)
LABEL = "BTC+ETH perpetual vs quarterly, 1h"

GRIDS = {
    "R6A": ("r6a_short_basis", [dict()]),
    "R6B": ("r6b_basis_vs_funding", [dict(theta=0.02), dict(theta=0.05)]),
}
assert sum(len(g[1]) for g in GRIDS.values()) == 3  # the pre-registered number of trials


@dataclass
class Venue:
    name: str
    bars: tuple  # open, high, low, close: hourly frames, columns = perpetuals + quarterlies
    funding: pd.DataFrame  # perpetual funding rates by settlement time, one column per perpetual
    legs: dict  # perpetual column -> list[Contract]
    fingerprint: str

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.bars[3].index


def _frames(raw: dict) -> tuple:
    idx = pd.DatetimeIndex(sorted(set().union(*(set(df["open_time"]) for df in raw.values()))))
    return tuple(pd.DataFrame({s: df.set_index("open_time")[f] for s, df in raw.items()}).reindex(idx)
                 for f in ("open", "high", "low", "close"))


def _listed(bars: pd.DataFrame) -> pd.Timestamp:
    """Listing = open time of the first bar with volume."""
    return pd.Timestamp(bars.loc[bars["volume"] > 0, "open_time"].iloc[0])


def load_binance() -> Venue:
    raw, legs, paths = {}, {}, []
    for perp, _ in COINS.values():
        raw[perp] = store.load_klines(perp, "1h", start=BARS_FROM)
        paths += [store.parquet_path("futures/um", store.KLINES, perp, "1h"),
                  store.parquet_path("futures/um", store.FUNDING, perp)]
        legs[perp] = []
        for sym in quarterly.stored_symbols(perp):
            k = quarterly.load_bars(sym)
            if k.empty or (k["volume"] > 0).sum() == 0:
                continue  # listed only inside the holdout
            raw[sym] = k
            legs[perp].append(Contract(sym, _listed(k), quarterly.delivery(sym)))
            paths.append(store.parquet_path("futures/um", store.KLINES, sym, "1h"))
    funding = pd.DataFrame({p: store.load_funding(p).set_index("funding_time")["rate"] for p in legs}).sort_index()
    return Venue("binance", _frames(raw), funding, legs, store.data_fingerprint(paths))


def deribit_names(coin: str) -> list[str]:
    return deribit.quarterly_names(coin, *DERIBIT_DELIVERIES)


def download_deribit() -> None:
    for coin, (_, perp) in COINS.items():
        print(deribit.build_candles(perp, BARS_FROM), flush=True)
        print(deribit.build_funding(perp, FUNDING_FROM), flush=True)
        for name in deribit_names(coin):
            print(deribit.build_candles(name, deribit.delivery(name) - pd.Timedelta(days=400)), flush=True)


def load_deribit() -> Venue:
    raw, legs, paths, funding = {}, {}, [], {}
    for coin, (_, perp) in COINS.items():
        k = deribit.load_candles(perp)
        raw[perp] = k[k["open_time"] >= BARS_FROM]
        f = deribit.load_funding(perp)
        funding[perp] = f.set_index("funding_time")["rate"]
        paths += [store.parquet_path(deribit.MARKET, store.KLINES, perp, "1h"),
                  store.parquet_path(deribit.MARKET, store.FUNDING, perp)]
        legs[perp] = []
        for name in deribit_names(coin):
            q = deribit.load_candles(name)
            q = q[q["open_time"] >= BARS_FROM]
            if q.empty or (q["volume"] > 0).sum() == 0:
                continue
            raw[name] = q
            legs[perp].append(Contract(name, _listed(q), deribit.delivery(name)))
            paths.append(store.parquet_path(deribit.MARKET, store.KLINES, name, "1h"))
    return Venue("deribit", _frames(raw), pd.DataFrame(funding).sort_index(), legs, store.data_fingerprint(paths))


def costs_for(v: Venue) -> CostModel:
    slip = {p: PERP_SLIPPAGE_BPS for p in v.legs}
    slip |= {c.symbol: QUARTERLY_SLIPPAGE_BPS for cs in v.legs.values() for c in cs}
    return CostModel(taker_fee=TAKER_FEE, default_slippage_bps=QUARTERLY_SLIPPAGE_BPS, slippage_bps=slip)


def build_weights(v: Venue, fam: str, theta: float | None = None, coins: list[str] | None = None):
    """Target weights for every column of the venue and the per-perpetual decision table."""
    C = v.bars[3]
    parts, feats = [], {}
    for perp, cs in v.legs.items():
        cols = [c.symbol for c in cs]
        f = basis.features(C.index, C[perp], C[cols], cs, v.funding[perp].dropna())
        s, c = basis.states_r6a(f) if fam == "R6A" else basis.states_r6b(f, theta)
        w = basis.weights(C.index, f, s, c, perp, cols)
        if coins is not None and perp not in coins:
            w[:] = np.nan  # this coin is not traded in a single-coin breakdown
        parts.append(w)
        feats[perp] = f.assign(state=s, contract=c)
    return pd.concat(parts, axis=1).reindex(columns=C.columns), feats


def check_execution(v: Venue, W: pd.DataFrame) -> None:
    """Both legs of every decision must have a bar at the execution bar; otherwise the engine would trade one leg
    only and leave the other unhedged. The data has no such gap; this makes sure it stays that way."""
    O = v.bars[0]
    rows = np.flatnonzero(W.notna().any(axis=1).to_numpy())
    for i in rows:
        cols = W.columns[W.iloc[i].notna()]
        if i + 1 >= len(O):
            continue  # a decision at the end of the data is never executed
        missing = [c for c in cols if not np.isfinite(O.iloc[i + 1][c]) and W.iloc[i][c] != 0]
        if missing:
            raise RuntimeError(f"no execution bar after {W.index[i]} for {missing}")


def rf_daily(index: pd.DatetimeIndex) -> pd.Series:
    from cqs.research.run_round2 import tbill
    rf = tbill()
    return rf.reindex(index.union(rf.index)).ffill().reindex(index)


def tbill_on_margin(res, rf: pd.Series) -> pd.Series:
    """T-bill yield forgone on the capital tied up as margin during each day (daily mean of the bar fraction)."""
    gross = res.positions.abs().sum(axis=1)
    tied = (MARGIN * gross / res.equity).clip(upper=1.0).resample("1D").mean()
    r = rf.reindex(tied.index).ffill().fillna(0.0)
    return tied * r / 365.0


def run_variant(v: Venue, W: pd.DataFrame, rf: pd.Series):
    """Daily excess returns at base and doubled costs, the per-variant summary and the full metrics."""
    check_execution(v, W)
    cm = costs_for(v)
    res = engine.run(*v.bars, W, cm, funding=v.funding)
    res2 = engine.run(*v.bars, W, cm.scaled(2), funding=v.funding)
    raw = metrics.daily_returns(res.equity, res.initial_equity)
    adj = tbill_on_margin(res, rf).reindex(raw.index).fillna(0.0)
    r = raw - adj
    raw2 = metrics.daily_returns(res2.equity, res2.initial_equity)
    r2 = raw2 - tbill_on_margin(res2, rf).reindex(raw2.index).fillna(0.0)
    m = metrics.compute(res) | {"sharpe_excess": walkforward.ann_sharpe(r), "total_excess_return": float((1 + r).prod() - 1),
                                "tbill_forgone": float(adj.sum())}
    pv = {k: m[k] for k in ("sharpe", "sharpe_excess", "cagr", "total_excess_return", "max_drawdown", "n_trades",
                            "exposure", "avg_gross_leverage", "turnover_per_year", "costs_paid", "funding_paid",
                            "tbill_forgone")}
    return r, r2, pv, m


def _decision_stats(feats: dict) -> dict:
    out = {}
    for perp, f in feats.items():
        s = f["state"].to_numpy()
        c = np.array([x if isinstance(x, str) else "" for x in f["contract"]], dtype=object)
        changed = (s[1:] != s[:-1]) | (c[1:] != c[:-1])
        rolls = (s[1:] == s[:-1]) & (s[1:] != 0) & (c[1:] != c[:-1])
        out[perp] = {"decisions": int(len(f)), "share_in_position": float(np.mean(s != 0)),
                     "share_short_basis": float(np.mean(s == 1)), "share_long_basis": float(np.mean(s == -1)),
                     "changes": int(changed.sum()), "rolls_in_position": int(rolls.sum()),
                     "days_without_front": int(f["front"].isna().sum())}
    return out


def run_family(fam: str, v: Venue, rf: pd.Series, dclose: pd.Series, log: bool = True):
    fname, grid = GRIDS[fam]
    daily, daily_x2, per_variant = {}, {}, {}
    for params in grid:
        name = _name(fname, params)
        W, feats = build_weights(v, fam, params.get("theta"))
        try:
            r, r2, pv, m = run_variant(v, W, rf)
        except RuntimeError as e:  # ruin or an execution gap is a result, not an error to hide
            per_variant[name] = {"ruined": str(e)}
            if log:
                registry.log_experiment(fam, name, params | {"market": LABEL, "venue": v.name}, per_variant[name],
                                        v.fingerprint, (str(BARS_FROM.date()), DEV_END))
            continue
        pv["decisions"] = _decision_stats(feats)
        daily[name], daily_x2[name], per_variant[name] = r, r2, pv
        if log:
            registry.log_experiment(fam, name, params | {"market": LABEL, "venue": v.name,
                                                         "measure": "excess over T-bills on margin"},
                                    m, v.fingerprint, (str(r.index[0].date()), str(r.index[-1].date())),
                                    sharpe_per_period=stats.sharpe_per_period(r.to_numpy()))
    out = evaluate_daily(fam, daily, daily_x2, per_variant, dclose, f"{v.name} {LABEL}", v.fingerprint, log=log,
                         keep_series=True, n_variants=len(grid), start=START)
    oos = out.pop("_oos")
    out.pop("_oos_x2", None)
    if len(grid) == 1:
        out["pbo_note"] = "one variant, no selection: PBO not applicable (pre-registered)"
    if len(oos):
        out["benchmarks_same_window"] = metrics_window.benchmarks_in_window(str(oos.index[0].date()), DEV_END, COSTS)
    return out, oos


def _stitch(daily: dict, choices: list) -> pd.Series:
    parts = []
    for ch in choices:
        r = daily[ch["variant"]]
        parts.append(r[(r.index >= pd.Timestamp(ch["test_start"], tz="UTC")) &
                       (r.index < pd.Timestamp(ch["test_end"], tz="UTC"))])
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def confirm(fam: str, choices: list, v: Venue, rf: pd.Series, log: bool = True) -> dict:
    """Same rules on the second venue, with the variant chosen on Binance in each walk-forward window (no tuning).
    Pre-registered condition: positive total excess return out of sample at base costs (BTC+ETH)."""
    fname, grid = GRIDS[fam]
    out = {"venue": v.name, "data": v.fingerprint}
    for label, coins in (("BTC+ETH", None), *((c, [p]) for c, (_, p) in COINS.items())):
        if coins is not None:
            coins = [p for p in v.legs if p in coins]
        daily, daily2, pvs = {}, {}, {}
        for params in grid:
            name = _name(fname, params)
            W, feats = build_weights(v, fam, params.get("theta"), coins)
            r, r2, pv, _ = run_variant(v, W, rf)
            pv["decisions"] = {k: d for k, d in _decision_stats(feats).items() if coins is None or k in coins}
            daily[name], daily2[name], pvs[name] = r, r2, pv
        oos, oos2 = _stitch(daily, choices), _stitch(daily2, choices)
        out[label] = {"oos": walkforward.summary(oos) if len(oos) else None,
                      "oos_costs_x2": walkforward.summary(oos2) if len(oos2) else None, "variants": pvs}
    total = (out["BTC+ETH"]["oos"] or {}).get("total_return")
    out["passed"] = bool(total is not None and np.isfinite(total) and total > 0)
    if log:
        registry.log_event("round6_confirmation", {"family": fam, "venue": v.name, "data": v.fingerprint,
                                                   "oos": out["BTC+ETH"]["oos"], "passed": out["passed"]})
    return out


def summarize(results: dict, oos: dict) -> dict:
    ex = [r for r in registry.read() if r.get("type") == "experiment" and r.get("family") != "BENCH"]
    n_all = len(ex)
    sr0 = null_sr0(n_all)
    fams = {}
    for fam, out in results.items():
        r = oos[fam].to_numpy()
        dsr = stats.deflated_sharpe(r, n_all, 1.0 / T_DAYS) if len(r) > 30 else float("nan")
        checks = checklist(out, dsr)
        if len(GRIDS[fam][1]) == 1:
            checks["pbo_le_0_25"] = True  # no selection within the family (pre-registered: not applicable)
        checks["deribit_confirmation"] = bool((out.get("deribit_confirmation") or {}).get("passed"))
        fams[fam] = {"oos": out["walk_forward"]["oos"], "oos_costs_x2": out["walk_forward"]["oos_costs_x2"],
                     "pbo": out["pbo"], "dsr_program_null_variance": dsr, "regimes_oos": out["regimes_oos"],
                     "benchmarks_same_window": out.get("benchmarks_same_window"),
                     "deribit_oos": (out.get("deribit_confirmation") or {}).get("BTC+ETH", {}).get("oos"),
                     "checks": checks, "candidate": all(checks.values())}
    n_oos = max((len(o) for o in oos.values()), default=0)
    needed = (sr0 + st.norm.ppf(0.95) / np.sqrt(n_oos - 1)) * np.sqrt(365) if n_oos > 1 else float("nan")
    return {"n_trials_program": n_all, "dsr_method": "null variance 1/T (as in round 5)", "T_days": T_DAYS,
            "sr0_annual": sr0 * np.sqrt(365), "oos_sharpe_for_dsr_0_95_annual_if_normal": needed,
            "n_oos_days": n_oos, "families": fams, "candidates": [f for f, v in fams.items() if v["candidate"]]}


def preregistered() -> bool:
    """The round-6 section must be in the committed pre-registration before any trial runs."""
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and "## Runda 6" in path.read_text(encoding="utf-8")


def _done(fam: str) -> bool:
    return registry.trial_count(fam) > 0


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["--download-deribit"]:
        download_deribit()
        return 0
    if not preregistered():
        raise SystemExit("round 6 is not pre-registered in a committed research/preregistration.md")
    fams = args or list(GRIDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v = vd = rf = dclose = None
    ran = False
    for fam in fams:
        if _done(fam):
            print(fam, "already in the registry: kept, not rerun", flush=True)
            continue
        if v is None:
            v, vd = load_binance(), load_deribit()
            rf = rf_daily(pd.date_range(FUNDING_FROM, pd.Timestamp(DEV_END, tz="UTC"), freq="1D"))
            dclose = store.load_klines("BTCUSDT", "1d").set_index("open_time")["close"]
        t0 = time.time()
        out, o = run_family(fam, v, rf, dclose)
        out["deribit_confirmation"] = confirm(fam, out["walk_forward"]["choices"], vd, rf)
        o.rename(fam).to_csv(RESULTS_DIR / f"{fam}_oos.csv")
        (RESULTS_DIR / f"{fam}.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
        print(fam, f"{time.time() - t0:.0f}s", "OOS:", out["walk_forward"]["oos"], "PBO:", out["pbo"],
              "Deribit passed:", out["deribit_confirmation"]["passed"], flush=True)
        ran = True
    if (ran or not (RESULTS_DIR / "R6_summary.json").exists()) and \
            all(_done(f) and (RESULTS_DIR / f"{f}_oos.csv").exists() for f in GRIDS):
        results = {f: json.loads((RESULTS_DIR / f"{f}.json").read_text()) for f in GRIDS}
        oos = {f: pd.read_csv(RESULTS_DIR / f"{f}_oos.csv", index_col=0, parse_dates=True).iloc[:, 0] for f in GRIDS}
        summ = summarize(results, oos)
        (RESULTS_DIR / "R6_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
        registry.log_event("round6_summary", {"candidates": summ["candidates"],
                                              "n_trials_program": summ["n_trials_program"]})
        print("candidates:", summ["candidates"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
