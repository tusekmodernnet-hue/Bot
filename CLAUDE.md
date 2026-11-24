# Crypto Quant System (cqs)

Research system for crypto perpetual futures strategies. The design and research protocol live in the
project's Stage 1 document; this file holds the rules every change must respect.

## Hard rules (set by the owner, never relax them)
1. Never present simulated results as real, never fabricate data or results.
2. No claim of an edge without the validation protocol below.
3. Never hide losing periods or unfavourable results.
4. No look-ahead: a decision at bar t uses only data available at the close of bar t; execution is at the
   next bar's open. Exchange metrics carry a 5-minute delay, off-exchange daily data a 1-day delay.
5. Always include fees, slippage and funding (`cqs.backtest.costs.CostModel`).
6. Universe is point-in-time and includes delisted contracts (`cqs.data.universe`).
7. The holdout (2025-06-01 onwards) is locked: `cqs.data.guard` refuses it unless a final test is declared.
   Never bypass the guard in research code.
8. Every experiment is logged in `research/registry.jsonl` (`cqs.research.registry`), including failures.
   The trial count feeds the Deflated Sharpe Ratio.
9. No real trading. API keys (later) are trade-only, without withdrawal rights, stored outside the repo.
10. No martingale, no averaging down, no raising risk to recover losses.

## Layout
- `src/cqs/data`: Binance public archive client, parsers, validation, Parquet store, universe, holdout guard
- `src/cqs/backtest`: cost model, engine (next-bar execution, stops, funding, delisting), metrics
- `src/cqs/research`: experiment registry, statistics (PSR, DSR, PBO)
- `tests`: unit tests on cases with known results. Run `python -m pytest`.

## Commands
- `pip install -e .[dev]` then `python -m pytest`
- `cqs download --symbols BTCUSDT ETHUSDT --interval 1h` downloads klines + funding into `data/`
- `cqs validate --symbols BTCUSDT --interval 1h` prints the data-quality report
