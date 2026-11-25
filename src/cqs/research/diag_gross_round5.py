"""Diagnostic, not a trial (nothing is logged in the registry): round-5 variants at zero fees and slippage, funding
kept. Tells whether a family has a gross edge that costs eat, or none at all.

Run: python -m cqs.research.diag_gross_round5 [R5A ...]  -> research/results/R5_gross_diagnostic.json
     python -m cqs.research.diag_gross_round5 --orb  -> adds the R5D breakdown ("_orb_breakdown")
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from cqs.backtest import engine, metrics
from cqs.backtest.costs import CostModel
from cqs.backtest.limit_sim import Fees, simulate
from cqs.research import registry, walkforward
from cqs.research import run_round5 as R
from cqs.research.run_ext import _name
from cqs.research.run_family import RESULTS_DIR
from cqs.strategies import orderflow as F

ZERO = CostModel(taker_fee=0.0, default_slippage_bps=0.0, forced_exit_slippage_bps=0.0)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")


def _summary(r: pd.Series, pnl: pd.Series, turnover: float) -> dict:
    o = r[r.index >= OOS_START]
    loss = -pnl[pnl <= 0].sum()
    return {"gross_sharpe_full": walkforward.ann_sharpe(r), "gross_sharpe_2022_2025": walkforward.ann_sharpe(o),
            "gross_total_return_2022_2025": float((1 + o).prod() - 1),
            "gross_profit_factor": float(pnl[pnl > 0].sum() / loss) if loss > 0 else float("nan"),
            "n_trades": int(len(pnl)), "turnover_per_year": turnover}


def main(argv=None) -> int:
    fams = (argv if argv is not None else sys.argv[1:]) or list(R.GRIDS)
    d = R.load_data()
    path = RESULTS_DIR / "R5_gross_diagnostic.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for fam in fams:
        fname, fn, grid = R.GRIDS[fam]
        for params in grid:
            name = _name(fname, params)
            if params.get("execution") == "limit":
                R.load_m1(d)
                rs, pnls, notional = [], [], 0.0
                for s in R.MAJORS:
                    m1 = d.m1[s]
                    plans = F.r5d_orb_limit_plans(d.m15[s], d.h1[s], m1.index, params["minutes"])
                    eq, t = simulate(m1.index, m1["open"], m1["high"], m1["low"], m1["close"], plans, Fees(0, 0, 0),
                                     d.funding[s].dropna())
                    rs.append(metrics.daily_returns(eq))
                    pnls.append(t["return"])
                r = sum(rs).dropna()
                out[name] = {"family": fam} | _summary(r, pd.concat(pnls, ignore_index=True), float("nan"))
            else:
                W, S = {}, {}
                for s in R.MAJORS:
                    w, stp = fn(d, s, **params)
                    W[s] = pd.Series(np.asarray(w, float), index=d.m15[s].index).reindex(d.index15)
                    S[s] = pd.Series(np.asarray(stp, float), index=d.m15[s].index).reindex(d.index15)
                res = engine.run(*d.bars, pd.DataFrame(W), ZERO, stops=pd.DataFrame(S), funding=d.funding,
                                 reenter_after_stop=True)
                r = metrics.daily_returns(res.equity)
                out[name] = {"family": fam} | _summary(r, res.trades.pnl, float(res.traded_notional / (len(r) / 365)))
            print(name, round(out[name]["gross_sharpe_full"], 2), flush=True)
    path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
    return 0


def orb_breakdown() -> dict:
    """Why R5D lost the gross edge of the R3E New York ORB (30 minutes, zero costs): sessions passing vs failing the
    volatility filter (both only where the filter is defined), and the market-entry result of filtered sessions
    split by whether the limit order at the edge was filled."""
    from cqs.strategies import intraday as I

    d = R.load_data()
    R.load_m1(d)
    out = {}
    W = {k: {} for k in ("pass", "fail")}
    S = {}
    filled_days = {}
    for s in R.MAJORS:
        m15, h1 = d.m15[s], d.h1[s]
        c, h, l = m15["close"].to_numpy(), m15["high"].to_numpy(), m15["low"].to_numpy()
        key, end_pos = I.session_keys(m15.index, "ny")
        v = F.ny_vol_values(m15, h1, key)
        ok = (v["value"] >= v["median_prev60"]).where(v["median_prev60"].notna())
        entry, stop = F._orb_signals(c, h, l, key, end_pos, 30)
        state = np.array([ok.get(k, np.nan) if k != -1 else np.nan for k in key], dtype=float)
        for name, want in (("pass", 1.0), ("fail", 0.0)):
            e = np.where(state == want, entry, 0.0)
            w, stp = I.trade_machine(c, h, l, e, stop, deadline=end_pos)
            W[name][s] = pd.Series(np.asarray(w, float), index=m15.index).reindex(d.index15)
            S[(name, s)] = pd.Series(np.asarray(stp, float), index=m15.index).reindex(d.index15)
        m1 = d.m1[s]
        plans = F.r5d_orb_limit_plans(m15, h1, m1.index, 30)
        _, t = simulate(m1.index, m1["open"], m1["high"], m1["low"], m1["close"], plans, Fees(0, 0, 0),
                        d.funding[s].dropna())
        filled_days[s] = set(pd.DatetimeIndex(t["entry_time"]).tz_convert(I.NY).normalize().tz_localize(None))
    for name in ("pass", "fail"):
        Wd = pd.DataFrame(W[name])
        Sd = pd.DataFrame({s: S[(name, s)] for s in R.MAJORS})
        res = engine.run(*d.bars, Wd, ZERO, stops=Sd, funding=d.funding, reenter_after_stop=True)
        r = metrics.daily_returns(res.equity)
        out[f"orb30_{name}_filter"] = _summary(r, res.trades.pnl, float("nan"))
        if name == "pass":
            t = res.trades
            day = pd.DatetimeIndex(t["entry_time"]).tz_convert(I.NY).normalize().tz_localize(None)
            hit = np.array([dd in filled_days[a] for dd, a in zip(day, t["asset"])])
            for lab, m in (("limit_filled", hit), ("limit_not_filled", ~hit)):
                roe = t.loc[m, "return_on_equity"]
                out[f"orb30_market_trades_{lab}"] = {"n": int(m.sum()), "mean_roe": float(roe.mean()),
                                                     "win_rate": float((roe > 0).mean())}
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--orb"]:
        path = RESULTS_DIR / "R5_gross_diagnostic.json"
        diag = json.loads(path.read_text()) if path.exists() else {}
        diag["_orb_breakdown"] = orb_breakdown()
        path.write_text(json.dumps(registry._clean(diag), indent=1, ensure_ascii=False))
        print(json.dumps(registry._clean(diag["_orb_breakdown"]), indent=1))
    else:
        raise SystemExit(main())
