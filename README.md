# Bot — system badawczy dla kontraktów perpetual na kryptowaluty

System bada, testuje i waliduje strategie na kontraktach perpetual. Handel realny jest wyłączony.

**Stan na 2026-11-26: badania trwają.** Właściciel polecił szukać dalej, aż do skutku. Osiem prerejestrowanych rund
(197 prób: dzienne, godzinowe, SMC/ICT i day trading na 15m, scalping na zleceniach limit na 1m, przepływ zleceń,
likwidacje, otwarcie sesji NY, baza kontraktów kwartalnych wobec fundingu, carry spot–perpetual na wielu monetach,
arbitraż statystyczny Avellanedy i Lee, short nowych tokenów i nowych kontraktów oraz wiek tokena) nie dało strategii,
która przeszłaby walidację. Najsilniejszy sygnał (R7C, short nowych tokenów: Sharpe OOS 1,01) odpada na obsunięciu
−60% i DSR 0,38, a jego wynik zależy od I kwartału 2025; z zabezpieczeniem altcoinami (R8A) Sharpe spada do 0,84.
Holdout (od 2025-06-01) nigdy nie został otwarty.
Wyniki: `research/results/`, rejestr prób: `research/registry.jsonl`, prerejestracje: `research/preregistration.md`.
Pełny opis danych, architektury, protokołu badawczego i limitów ryzyka znajduje się w dokumencie projektu
„System Quant Krypto — Etap 1”. Zasady, których kod nie może łamać, są w [CLAUDE.md](CLAUDE.md).

## Stan prac

| Etap | Stan |
| --- | --- |
| 1. Research danych, architektura, protokół | Gotowe |
| 2. Fundament: dane, magazyn, backtest, statystyka, rejestr | Gotowe, 124 testy przechodzą |
| 3. Odkrywanie i walidacja strategii (H1–H7, ML, rundy 2–8) | W toku: 197 prób w ośmiu rundach, brak kandydata |
| 4. Paper trading i raport końcowy | Nie startuje: brak strategii, która przeszła walidację |

## Szybki start

```bash
pip install -e ".[dev]"
python -m pytest                                    # testy na przypadkach o znanym wyniku
cqs symbols > symbols.txt                           # wszystkie kontrakty USDT, łącznie ze zdjętymi z giełdy
cqs download --symbols BTCUSDT ETHUSDT --interval 1h --metrics
cqs validate --symbols BTCUSDT ETHUSDT --interval 1h
```

Dane trafiają do `data/` i nie są commitowane. Każdy plik z archiwum jest weryfikowany sumą SHA-256.

## Moduły

| Moduł | Rola |
| --- | --- |
| `cqs.data.archive` | Klient publicznego archiwum Binance: listing, pobieranie, sumy kontrolne, cache |
| `cqs.data.parsers` | Parsowanie plików z nagłówkiem i bez, znaczniki czasu w ms i µs |
| `cqs.data.validate` | Luki, duplikaty, błędne świece OHLC i kontrakty-zombie po delistingu |
| `cqs.data.store` | Magazyn Parquet + DuckDB, manifest źródeł, odcisk danych dla eksperymentów |
| `cqs.data.guard` | Blokada holdoutu (od 2025-06-01), otwierana tylko jawnie i z wpisem w rejestrze |
| `cqs.data.universe` | Uniwersum point-in-time: wiek, aktywny obrót, płynność, kontrakty zdjęte z giełdy |
| `cqs.data.spotmap` | Dopasowanie perpetual → para spot (z mnożnikiem kontraktu), daty pierwszych notowań, pobranie danych rundy 7 |
| `cqs.data.quarterly`, `cqs.data.deribit` | Kontrakty kwartalne Binance (dostawa, odcięcie rekordów po dostawie) i dane Deribit (świece 1h, funding godzinowy) |
| `cqs.backtest.engine` | Wykonanie na otwarciu następnej świecy, koszty, funding, stopy, wymuszone zamknięcia, pasmo rebalansu od kapitału albo od wagi docelowej |
| `cqs.backtest.limit_sim` | Konserwatywna symulacja zleceń limit na świecach 1m (wypełnienie tylko przy przejściu ceny) |
| `cqs.backtest.costs` | Model kosztów: prowizja, slippage, test warunków skrajnych (koszty ×2) |
| `cqs.backtest.metrics` | Sharpe, Sortino, Calmar, obsunięcia i ich czas, CAGR, metryki transakcji, ryzyko ogona |
| `cqs.research.stats` | Probabilistic i Deflated Sharpe Ratio, PBO metodą CSCV |
| `cqs.research.registry` | Rejestr eksperymentów `research/registry.jsonl`, w tym nieudanych |
| `cqs.research.benchmarks` | Benchmarki, które musi pobić każda strategia |
| `cqs.research.run_family`, `run_ext` | Pełna ocena rodziny: warianty, walk-forward, PBO, DSR, reżimy, koszty ×2 |
| `cqs.research.ml_meta` | Meta-model LightGBM z purgingiem i embargo oraz jego punkt odniesienia |
| `cqs.research.run_round3` … `run_round9` | Rundy 3–9 (od rundy 5: DSR z wariancją estymatora 1/T; od rundy 6: zwrot ponad bony od depozytu; runda 6: potwierdzenie na Deribit) |
| `cqs.research.diag_gross`, `diag_gross_round5` | Diagnostyka wyników przed kosztami rund 3 i 5 (nie jest próbą) |
| `cqs.strategies` | Prerejestrowane rodziny H1–H7, rundy 2–6 (`intraday`: SMC/ICT, ORB, VWAP; `scalping`: plany zleceń limit; `orderflow`: przepływ agresora, OI, ORB NY z filtrem; `basis`: kontrakt frontowy, baza, funding; `carry_multi`, `statarb`, `listings`: runda 7; `supply`: runda 8; `listing_risk`: runda 9) |

## Zasady czasu (brak look-ahead)

- Decyzja z zamknięcia świecy *t* jest wykonywana na otwarciu świecy *t+1*.
- Dane metrics (OI, long/short) są dostępne 5 minut po znaczniku czasu, dane dzienne spoza giełdy 1 dzień później.
- Funding obciąża pozycję utrzymywaną w chwili rozliczenia: long płaci dodatnią stawkę, short ją otrzymuje.
- Stop wypełnia się po cenie stopu, a przy luce po gorszym otwarciu.
- Pozycja w kontrakcie, który przestał być notowany, jest zamykana po ostatniej cenie z 2% dodatkowego poślizgu.
