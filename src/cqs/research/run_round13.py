"""Round 13 (research/preregistration.md, "Runda 13"): R9A's rule on new Hyperliquid perpetuals, one trial on daily
bars, data before 2025-06-01 only.

R13A  events = the first trading day D0 of each Hyperliquid perpetual, except BTC and old tokens (a Binance USDT
      perpetual listed more than 30 days before D0); shorted at the close of D0 + 1 if that day's volume is at least
      2M USD, -0.05 of equity each, at most 20 at a time, 30 days in position, close stop at 2x the entry open, hedged
      with the top-20 Hyperliquid alt basket for the same notional. Taker 0.045%, slippage 20 bp, Hyperliquid funding.
The rule was fixed in round 9 on Binance data, so there is no walk-forward: the whole Hyperliquid period is out of
sample for it. Measured in excess of T-bills on the margin (50% of gross notional, at most the equity).
R13A is a replication, never a candidate; the pre-registered verdict is replicated / not replicated / inconclusive.

Run: python -m cqs.data.hyperliquid, then python -m cqs.research.run_round13
     -> research/results/R13A.json, R13A_daily.csv, R13_summary.json
A family already in the registry is never rerun. The runner refuses to start unless the round-13 section of the
pre-registration is committed, and refuses traded contracts without funding data.
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd

from cqs import config
from cqs.backtest import engine, metrics
from cqs.backtest.costs import CostModel
from cqs.data import hyperliquid, store
from cqs.research import registry, stats, walkforward
from cqs.research import run_round7 as R7
from cqs.research.run_ext import _name
from cqs.research.run_family import RESULTS_DIR
from cqs.strategies import hl_listings, listing_risk

FAMILY = "R13A"
NAME = _name("r13a_hyperliquid_listing_shorts", dict(stop=1.0))
PARAMS = dict(stop=1.0)
STOP_K = 1.0
BAND_REL = 0.20
EVENTS_UNTIL = pd.Timestamp("2025-05-29", tz="UTC")
COSTS = CostModel(taker_fee=0.00045, default_slippage_bps=20.0)
MIN_SHORTS = 30
MAX_MISSING_SHARE = 0.10
PSR_MIN = 0.90
BINANCE_FIRST = config.REPO_ROOT / "research" / "data" / "binance_perp_first_day.csv"
LABEL = ("Hyperliquid new perpetuals (no Binance perp > 30 days earlier): shorts -0.05, max 20, d=1, H=30, "
         "volume >= 2M USD, close stop 2x + top20 Hyperliquid alt basket long (BTC excluded), 1d")


# ------------------------------------------------------------------ data
def write_binance_first_days() -> int:
    """research/data/binance_perp_first_day.csv from the research data (first traded day of each USDT perpetual)."""
    first_perp, _ = R7.listing_dates()
    rows = sorted((s, d.date().isoformat()) for s, d in first_perp.items()
                  if d is not None and re.fullmatch(r"[A-Z0-9]+USDT", s))
    BINANCE_FIRST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["symbol", "first_traded_day"]).to_csv(BINANCE_FIRST, index=False)
    return len(rows)


def binance_first_days() -> dict[str, pd.Timestamp]:
    df = pd.read_csv(BINANCE_FIRST)
    return {s: pd.Timestamp(d, tz="UTC") for s, d in zip(df["symbol"], df["first_traded_day"])}


def funding_hourly(fr: pd.DataFrame) -> pd.Series:
    """Settlement times rounded to the hour plus one second: Hyperliquid stamps each hourly settlement a few ms off
    the hour, and one shared index keeps the frame small. The second keeps a settlement just after the bar open."""
    t = fr["funding_time"].dt.round("h") + pd.Timedelta(seconds=1)
    return fr.groupby(t)["rate"].sum()


def frames(coins: list[str], with_funding: bool = True) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Daily OHLC + volumes on one complete UTC day index, and the funding frame (settlement x coin)."""
    cols = {k: {} for k in ("open", "high", "low", "close", "quote_volume", "volume")}
    fund = {}
    for c in coins:
        k = hyperliquid.load_candles(c)
        if k.empty:
            continue
        k = k.assign(open_time=k["open_time"].dt.normalize()).drop_duplicates("open_time", keep="last")
        k = k.set_index("open_time")
        traded = k.index[k["volume"] > 0]
        if not len(traded):
            continue
        k = k.loc[traded.min():traded.max()]  # no flat bars before the first or after the last trade
        for f in cols:
            cols[f][c] = k[f]
        if not with_funding:
            continue
        try:
            fr = hyperliquid.load_funding(c)
        except FileNotFoundError:
            continue
        if len(fr):
            fund[c] = funding_hourly(fr)
    out = {f: pd.DataFrame(v).sort_index() for f, v in cols.items()}
    idx = pd.date_range(out["close"].index.min(), out["close"].index.max(), freq="1D")
    out = {f: df.reindex(idx) for f, df in out.items()}
    funding = pd.DataFrame(fund).sort_index() if fund else pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    return out, funding


def load_r13a() -> dict:
    meta = json.loads(hyperliquid.meta_path().read_text())["universe"]
    summary = json.loads(hyperliquid.summary_path().read_text())
    coins = [a["name"] for a in meta
             if store.parquet_path(hyperliquid.MARKET, store.KLINES, a["name"], "1d").exists()]
    candles = {c: hyperliquid.load_candles(c) for c in coins}
    first_hl = hl_listings.first_traded_days(candles)
    bfirst = binance_first_days()
    events, why = hl_listings.events(first_hl, bfirst, EVENTS_UNTIL)
    fr_all, _ = frames(list(first_hl), with_funding=False)
    basket = hl_listings.basket(fr_all["close"], fr_all["quote_volume"], fr_all["volume"])
    need = list(dict.fromkeys(list(events.index) + list(basket.columns) + [hl_listings.HEDGE]))
    fr, funding = frames(need)
    basket = basket.reindex(index=fr["close"].index, columns=fr["close"].columns, fill_value=False).fillna(False)
    paths = [store.parquet_path(hyperliquid.MARKET, store.KLINES, c, "1d") for c in need] + \
            [store.parquet_path(hyperliquid.MARKET, store.FUNDING, c) for c in need]
    return dict(fr=fr, funding=funding, events=events, excluded={k: len(v) for k, v in why.items()},
                excluded_coins=why, basket=basket.astype(bool), rf=R7.rf_daily(fr["close"].index), costs=COSTS,
                fingerprint=R7._fp(paths), binance_first=bfirst, download=summary,
                near_misses=hl_listings.near_misses(events.index, set(bfirst)),
                missing_funding=[c for c in need if c not in funding.columns])


# ------------------------------------------------------------------ run
def r13a_weights(d: dict):
    fr = d["fr"]
    W, held, stops = listing_risk.r9a(fr["open"], fr["close"], fr["quote_volume"], d["events"], d["basket"], STOP_K,
                                      min_qv=hl_listings.MIN_QUOTE_VOLUME)
    return W.reindex(columns=fr["close"].columns, fill_value=0.0), held, stops


def excess_daily(res, rf: pd.Series) -> pd.Series:
    r = metrics.daily_returns(res.equity, res.initial_equity)
    return r - R7.tbill_on_margin(res, rf).reindex(r.index).fillna(0.0)


def verdict(sharpe: float, psr: float, sharpe_x2: float, n_shorts: int, missing_share: float) -> str:
    """Pre-registered: survivorship risk and too few shorts come first, then the replication rule."""
    if missing_share > MAX_MISSING_SHARE:
        return "survivorship_risk"
    if n_shorts < MIN_SHORTS:
        return "too_few_shorts"
    if not np.isfinite(sharpe) or sharpe <= 0:
        return "not_replicated"
    if psr >= PSR_MIN and np.isfinite(sharpe_x2) and sharpe_x2 > 0:
        return "replicated"
    return "inconclusive"


def by_year(r: pd.Series) -> dict:
    return {str(y): float((1 + x).prod() - 1) for y, x in r.groupby(r.index.year)}


def r9a_correlation(r: pd.Series) -> dict:
    path = RESULTS_DIR / "R9A_oos.csv"
    if not path.exists():
        return {"available": False}
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    a = pd.concat([r.rename("r13a"), s.rename("r9a")], axis=1, join="inner").dropna()
    return {"available": True, "common_days": int(len(a)),
            "correlation": float(a["r13a"].corr(a["r9a"])) if len(a) > 30 else None}


def trade_split(trades: pd.DataFrame, events: pd.Series, bfirst: dict) -> dict:
    """Diagnostics: short trades by whether the coin's Binance USDT perpetual already traded at the decision."""
    bset = set(bfirst)
    out = {"binance_perp_at_decision": {"n": 0, "pnl": 0.0}, "no_binance_perp_at_decision": {"n": 0, "pnl": 0.0}}
    for t in trades[trades["side"] == "short"].itertuples(index=False):
        if t.asset not in events.index:
            continue
        b = hl_listings.binance_symbol(t.asset, bset)
        decided = pd.Timestamp(events[t.asset]) + pd.Timedelta(days=listing_risk.DELAY)
        key = "binance_perp_at_decision" if b is not None and bfirst[b] <= decided else "no_binance_perp_at_decision"
        out[key]["n"] += 1
        out[key]["pnl"] += float(t.pnl)
    return out


def evaluate(d: dict, log: bool = True) -> tuple[dict, pd.Series]:
    fr = d["fr"]
    W, held, stops = r13a_weights(d)
    traded = [c for c in W.columns if bool((W[c] != 0).any())]
    missing = [c for c in traded if c not in d["funding"].columns]
    if missing:
        raise SystemExit(f"R13A: no funding data for {len(missing)} traded contracts, e.g. {missing[:5]}")
    bars = (fr["open"], fr["high"], fr["low"], fr["close"])
    res = engine.run(*bars, W, d["costs"], funding=d["funding"], rebalance_band_rel=BAND_REL)
    res2 = engine.run(*bars, W, d["costs"].scaled(2), funding=d["funding"], rebalance_band_rel=BAND_REL)
    r_all, r2_all = excess_daily(res, d["rf"]), excess_daily(res2, d["rf"])
    gross = res.positions.abs().sum(axis=1).resample("1D").last()
    first = gross[gross > 0].index.min() if (gross > 0).any() else None
    r = r_all.loc[first:] if first is not None else r_all.iloc[:0]
    r2 = r2_all.loc[first:] if first is not None else r2_all.iloc[:0]
    m = metrics.compute(res)
    m["sharpe_excess"] = walkforward.ann_sharpe(r)
    m["total_excess_return"] = float((1 + r).prod() - 1) if len(r) else float("nan")
    period = (str(fr["close"].index[0].date()), str(fr["close"].index[-1].date()))
    if log:
        registry.log_experiment(FAMILY, NAME, PARAMS | {"market": LABEL}, m, d["fingerprint"], period,
                                sharpe_per_period=stats.sharpe_per_period(r.to_numpy()) if len(r) > 2 else None)
    n_prog = len([x for x in registry.read() if x.get("type") == "experiment" and x.get("family") != "BENCH"])
    arr = r.to_numpy()
    psr = stats.probabilistic_sharpe(arr, 0.0) if len(arr) > 30 else float("nan")
    dsr = stats.deflated_sharpe(arr, n_prog, 1.0 / len(arr)) if len(arr) > 30 else float("nan")
    shorts = int(sum(1 for c in held.columns if bool(held[c].any())))
    summ = walkforward.summary(r) if len(r) else None
    summ2 = walkforward.summary(r2) if len(r2) else None
    btc = fr["close"][hl_listings.HEDGE] if hl_listings.HEDGE in fr["close"] else None
    reg = walkforward.regimes(btc) if btc is not None else None
    miss = d["download"].get("missing_share", 0.0)
    out = {"family": FAMILY, "variant": NAME, "market": LABEL, "data": d["fingerprint"], "period": period,
           "measure": "excess over T-bills on margin", "costs": {"taker_fee": d["costs"].taker_fee,
                                                                "slippage_bps": d["costs"].default_slippage_bps},
           "rebalance": f"band {BAND_REL:.0%} of each target notional; entries and exits always trade",
           "events": int(len(d["events"])), "excluded": d["excluded"], "near_misses": d["near_misses"],
           "shorts_traded": shorts, "stops": listing_risk.stop_stats(stops, fr["open"]),
           "mean_shorts": float(held.sum(axis=1).loc[first:].mean()) if first is not None else 0.0,
           "download": {k: v for k, v in d["download"].items() if k != "missing"},
           "missing_coins": d["download"].get("missing", []),
           "excess": summ, "excess_costs_x2": summ2, "by_year": by_year(r) if len(r) else {},
           "regimes": R7._regime_ok(walkforward.by_regime(r, reg)) if reg is not None and len(r) else None,
           "psr_vs_0": psr, "dsr_program_null_variance": dsr, "n_trials_program": n_prog,
           "engine_metrics": {k: m[k] for k in ("sharpe", "cagr", "max_drawdown", "n_trades", "win_rate",
                                                "profit_factor", "exposure", "turnover_per_year", "costs_paid",
                                                "funding_paid")},
           "r9a_correlation": r9a_correlation(r), "trade_split": trade_split(res.trades, d["events"],
                                                                             d["binance_first"])}
    out["verdict"] = verdict(summ["sharpe"] if summ else float("nan"), psr,
                             summ2["sharpe"] if summ2 else float("nan"), shorts, miss)
    return out, r


# ------------------------------------------------------------------ main
def preregistered() -> bool:
    path = RESULTS_DIR.parent / "preregistration.md"
    committed = registry._git("status", "--porcelain", "--", str(path)) == ""
    return committed and "## Runda 13" in path.read_text(encoding="utf-8")


def _done() -> bool:
    return registry.trial_count(FAMILY) > 0


def main(argv=None) -> int:
    if not preregistered():
        raise SystemExit("round 13 is not pre-registered in a committed research/preregistration.md")
    if _done():
        print(FAMILY, "already in the registry: kept, not rerun", flush=True)
        return 0
    if not hyperliquid.summary_path().exists():
        raise SystemExit("no Hyperliquid data: run python -m cqs.data.hyperliquid first")
    t0 = time.time()
    d = load_r13a()
    out, r = evaluate(d)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    r.rename(FAMILY).to_csv(RESULTS_DIR / f"{FAMILY}_daily.csv")
    (RESULTS_DIR / f"{FAMILY}.json").write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
    summ = {"family": FAMILY, "verdict": out["verdict"], "excess": out["excess"],
            "excess_costs_x2": out["excess_costs_x2"], "psr_vs_0": out["psr_vs_0"],
            "dsr_program_null_variance": out["dsr_program_null_variance"], "n_trials_program": out["n_trials_program"],
            "events": out["events"], "shorts_traded": out["shorts_traded"], "candidate": False,
            "note": "replication of R9A on another venue; never a candidate (pre-registered)"}
    (RESULTS_DIR / "R13_summary.json").write_text(json.dumps(registry._clean(summ), indent=1, ensure_ascii=False))
    registry.log_event("round13_summary", {"verdict": out["verdict"], "n_trials_program": out["n_trials_program"]})
    print(FAMILY, f"{time.time() - t0:.0f}s", "verdict:", out["verdict"], "excess:", out["excess"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
