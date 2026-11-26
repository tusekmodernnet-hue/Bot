"""Diagnostic, not a trial (nothing is logged in the registry): how large an account R9A can carry before market
impact on its short legs exceeds the slippage the backtest charges. Research data only, the holdout stays closed.

For every trade on the short legs of R9A (k = 1.0, the variant of the pre-registered final test) the square-root
impact model estimates the cost at a given account size:
    impact = Y * sigma * sqrt(Q / V),  Y = 0.7,
sigma = Parkinson volatility of the trade day (ln(H/L) / sqrt(4 ln 2)), V = the contract's quote volume that day,
Q = |traded notional| / equity at the previous close x the account size. The backtest charges 10 bp of slippage on
these contracts (20 bp in the costs x2 test). Reported: the notional-weighted mean impact, quantiles for entries and
exits, and the yearly drag above the 10 bp already charged. The model assumes the order is worked through the day,
not sent as one market order at the open. Basket legs (top-20 alts) keep the backtest's cost assumption.

Run: python -m cqs.research.diag_capacity_r9a  -> research/results/R9A_capacity_diagnostic.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cqs.backtest import engine
from cqs.research import registry
from cqs.research.run_family import RESULTS_DIR

Y = 0.7
CHARGED = 10e-4
EQUITIES = (1e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2e6, 5e6)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
RESEARCH_END = pd.Timestamp("2025-06-01", tz="UTC")
ENTRY_EXIT = ("entry", "exit", "stop", "forced_exit")


def parkinson(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    return np.log(high / low) / np.sqrt(4 * np.log(2))


def impact(w, equity: float, qv, sigma, y: float = Y):
    """Square-root impact as a fraction of price for a trade of w x equity against a day's quote volume qv."""
    return y * sigma * np.sqrt(w * equity / qv)


def short_trades(fills: pd.DataFrame, W: pd.DataFrame, positions: pd.DataFrame, equity: pd.Series,
                 quote_volume: pd.DataFrame, sigma: pd.DataFrame, assets) -> pd.DataFrame:
    """One row per fill on a short leg: kind (entry, exit, rebalance, or the engine's stop / forced_exit), w =
    |notional| / equity at the previous close, and the day's quote volume and sigma. A fill at row i executes the
    target weight of row i - 1. Fills on a contract that is long at the time (a basket member) are left out."""
    ix = {t: i for i, t in enumerate(positions.index)}
    rows = []
    sel = fills.loc[fills["asset"].isin(list(assets)), ["time", "asset", "notional", "reason"]]
    for t, a, notional, reason in sel.itertuples(index=False):
        i = ix[t]
        if i == 0:
            continue
        tgt, before = float(W[a].iloc[i - 1]), float(positions[a].iloc[i - 1])
        if not (tgt < 0 or before < 0):
            continue
        if reason != "signal":
            kind = reason
        elif before == 0:
            kind = "entry"
        elif tgt == 0:
            kind = "exit"
        else:
            kind = "rebalance"
        rows.append((t, a, kind, abs(notional) / float(equity.iloc[i - 1]), quote_volume.at[t, a], sigma.at[t, a]))
    return pd.DataFrame(rows, columns=["time", "asset", "kind", "w", "qv", "sigma"])


def summarize(trades: pd.DataFrame, equities=EQUITIES, end: pd.Timestamp = RESEARCH_END,
              oos_start: pd.Timestamp = OOS_START) -> dict:
    ok = trades["qv"].gt(0) & trades["sigma"].gt(0)
    T = trades[ok]
    out = {"model": f"impact = {Y} x sigma(Parkinson, trade day) x sqrt(Q / quote volume of the trade day)",
           "charged_slippage_bp": CHARGED * 1e4, "dropped_no_volume_or_range": int((~ok).sum()),
           "by_kind": {k: {"n": int(len(S)), "median_w": float(S["w"].median()),
                           "median_quote_volume": float(S["qv"].median()),
                           "p10_quote_volume": float(S["qv"].quantile(0.1)),
                           "median_sigma": float(S["sigma"].median())} for k, S in T.groupby("kind")}}
    for wname, start in (("full", T["time"].min()), ("oos", oos_start)):
        S = T[(T["time"] >= start) & (T["time"] < end)]
        years = (end - start).days / 365.25
        ee = S["kind"].isin(ENTRY_EXIT)
        res = {"start": str(start.date()), "years": years, "n_trades": int(len(S)),
               "short_leg_turnover_per_year": float(S["w"].sum() / years), "by_equity": {}}
        for E in equities:
            imp = impact(S["w"], E, S["qv"], S["sigma"])
            res["by_equity"][f"{E:.0f}"] = {
                "weighted_mean_impact_bp": float(np.average(imp, weights=S["w"]) * 1e4),
                "drag_above_charged_per_year": float(((imp - CHARGED) * S["w"]).sum() / years),
                "entries_exits_median_bp": float(imp[ee].median() * 1e4),
                "entries_exits_p90_bp": float(imp[ee].quantile(0.9) * 1e4)}
        out[wname] = res
    return out


def main(argv=None) -> int:
    from cqs.research import run_round9 as R9
    from cqs.strategies import listing_risk
    d = R9.load_r9a()
    p = d["p"]
    W, held, _ = listing_risk.r9a(p.open, p.close, p.quote_volume, d["events"], d["basket"], 1.0)
    W = W.reindex(columns=p.close.columns, fill_value=0.0)
    res = engine.run(p.open, p.high, p.low, p.close, W, d["costs"], funding=p.funding,
                     rebalance_band_rel=R9.BAND_REL)
    T = short_trades(res.fills, W, res.positions, res.equity, p.quote_volume, parkinson(p.high, p.low), held.columns)
    out = {"diagnostic": "R9A capacity (not a trial; research data only, holdout closed)",
           "fingerprint": d["fingerprint"]} | summarize(T)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "R9A_capacity_diagnostic.json").write_text(json.dumps(registry._clean(out), indent=1))
    for E, v in out["oos"]["by_equity"].items():
        print(f"{float(E):>12,.0f} USD: mean impact {v['weighted_mean_impact_bp']:.1f} bp, drag above 10 bp "
              f"{v['drag_above_charged_per_year']:+.2%}/yr, entries+exits p90 {v['entries_exits_p90_bp']:.1f} bp",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
