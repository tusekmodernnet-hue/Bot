"""Command line entry point: `cqs download`, `cqs validate`, `cqs universe`."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys

import pandas as pd

from cqs.data import archive, store
from cqs.data.validate import check_bars


def cmd_download(a: argparse.Namespace) -> int:
    jobs = []
    for s in a.symbols:
        jobs.append((a.market, store.KLINES, s, a.interval))
        if a.funding:
            jobs.append((a.market, store.FUNDING, s, None))
        if a.metrics:
            jobs.append((a.market, store.METRICS, s, None))
    failed = 0
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        futs = {ex.submit(store.build, *j, workers=a.workers): j for j in jobs}
        for f in cf.as_completed(futs):
            j = futs[f]
            try:
                print("ok  ", "/".join(x for x in j if x), "->", f.result())
            except Exception as e:  # report and continue; failures are listed, never hidden
                failed += 1
                print("FAIL", "/".join(x for x in j if x), type(e).__name__, e, file=sys.stderr)
    print(f"{len(jobs) - failed} built, {failed} failed")
    return 1 if failed else 0


def cmd_validate(a: argparse.Namespace) -> int:
    for s in a.symbols:
        path = store.parquet_path(a.market, store.KLINES, s, a.interval)
        bars = pd.read_parquet(path)  # full stored history: quality checks may read the holdout
        print(json.dumps(check_bars(bars, s, a.interval).as_dict(), indent=1))
    return 0


def cmd_symbols(a: argparse.Namespace) -> int:
    syms = archive.list_symbols(a.market)
    syms = [s for s in syms if s.endswith(a.quote)]
    print("\n".join(syms))
    print(f"{len(syms)} symbols", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cqs")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="build Parquet datasets from the Binance archive")
    d.add_argument("--symbols", nargs="+", required=True)
    d.add_argument("--interval", default="1h")
    d.add_argument("--market", default="futures/um")
    d.add_argument("--no-funding", dest="funding", action="store_false")
    d.add_argument("--metrics", action="store_true", help="also build 5-minute OI/positioning metrics")
    d.add_argument("--parallel", type=int, default=4, help="symbols built at once")
    d.add_argument("--workers", type=int, default=16, help="downloads per symbol at once")
    d.set_defaults(func=cmd_download)
    v = sub.add_parser("validate", help="data-quality report for stored bars")
    v.add_argument("--symbols", nargs="+", required=True)
    v.add_argument("--interval", default="1h")
    v.add_argument("--market", default="futures/um")
    v.set_defaults(func=cmd_validate)
    s = sub.add_parser("symbols", help="list archive symbols, including delisted ones")
    s.add_argument("--market", default="futures/um")
    s.add_argument("--quote", default="USDT")
    s.set_defaults(func=cmd_symbols)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
