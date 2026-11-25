"""Diagnostic, not a trial (nothing is logged in the registry): round-3 variants at zero fees and slippage, funding
kept. Tells whether a family has a gross edge that costs eat, or none at all.

Run: python -m cqs.research.diag_gross [R3A ...]  -> research/results/R3_gross_diagnostic.json
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from cqs.backtest import engine, metrics
from cqs.backtest.costs import CostModel
from cqs.research import registry, walkforward
from cqs.research import run_round3 as R
from cqs.research.run_family import RESULTS_DIR

ZERO = CostModel(taker_fee=0.0, default_slippage_bps=0.0, forced_exit_slippage_bps=0.0)


def main(argv=None) -> int:
    fams = (argv if argv is not None else sys.argv[1:]) or list(R.GRIDS)
    d = R.load_data()
    path = RESULTS_DIR / "R3_gross_diagnostic.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for fam in fams:
        runs, _ = R.build_runs(fam, d)
        for name, _params, W, S in runs:
            res = engine.run(*d.bars, W, ZERO, stops=S, funding=d.funding, reenter_after_stop=True)
            r = metrics.daily_returns(res.equity)
            t = res.trades
            o = r[r.index >= pd.Timestamp("2022-01-01", tz="UTC")]
            out[name] = {"family": fam, "gross_sharpe_full": walkforward.ann_sharpe(r),
                         "gross_profit_factor": float(t.pnl[t.pnl > 0].sum() / -t.pnl[t.pnl <= 0].sum()),
                         "gross_sharpe_2022_2025": walkforward.ann_sharpe(o),
                         "gross_total_return_2022_2025": float((1 + o).prod() - 1), "n_trades": int(len(t)),
                         "turnover_per_year": float(res.traded_notional / (len(r) / 365))}
            print(name, round(out[name]["gross_sharpe_full"], 2), flush=True)
    path.write_text(json.dumps(registry._clean(out), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
