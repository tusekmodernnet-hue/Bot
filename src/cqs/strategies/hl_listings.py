"""Round 13 (research/preregistration.md, "Runda 13"): the events and the hedge basket of R13A, R9A's rule on new
Hyperliquid perpetuals.

An event is the first day D0 with a daily candle with volume of a Hyperliquid perpetual, unless the coin is BTC or
an old token: one whose Binance USDⓈ-M USDT perpetual first traded more than 30 days before D0 (the perpetual market
already existed, so R9A's mechanism does not apply). Names map to Binance as name + USDT; a leading k means 1000 units
(kPEPE -> 1000PEPEUSDT, else PEPEUSDT). Binance contracts that renamed an older token (`listings.RENAMES`) are old
tokens. The basket is the point-in-time top 20 Hyperliquid alts: traded for at least 90 days, 30-day mean volume of at
least 2M USD, ranked by that volume, BTC excluded.
"""

from __future__ import annotations

import pandas as pd

from cqs.data.universe import UniverseRules, eligible
from cqs.strategies import listings

OLD_LEAD = pd.Timedelta(days=30)
MIN_QUOTE_VOLUME = 2e6
HEDGE = "BTC"
BASKET_RULES = UniverseRules(min_age_days=90, liquidity_window_days=30, min_avg_quote_volume=MIN_QUOTE_VOLUME,
                             top_n=20, quote_asset="")


def binance_symbol(coin: str, binance: set[str]) -> str | None:
    """The Binance USDT perpetual of a Hyperliquid coin, if one exists in `binance`."""
    cands = []
    if len(coin) > 1 and coin[0] == "k" and coin[1].isupper():
        cands += [f"1000{coin[1:]}USDT", f"{coin[1:]}USDT"]
    cands.append(f"{coin.upper()}USDT")
    return next((c for c in cands if c in binance), None)


def first_traded_days(candles: dict[str, pd.DataFrame]) -> dict[str, pd.Timestamp]:
    """coin -> first UTC day with a daily candle with volume (coins without one are left out)."""
    out = {}
    for coin, k in candles.items():
        traded = k[k["volume"] > 0]
        if len(traded):
            out[coin] = pd.Timestamp(traded["open_time"].iloc[0]).normalize()
    return out


def events(first_hl: dict[str, pd.Timestamp], binance_first: dict[str, pd.Timestamp],
           until: pd.Timestamp) -> tuple[pd.Series, dict]:
    """coin -> D0 for every R13A event with D0 <= until, and the excluded coins by reason."""
    bset = set(binance_first)
    ev, why = {}, {"btc": [], "old_token": [], "renamed_token": [], "after_window": []}
    for coin, d0 in sorted(first_hl.items()):
        if coin == HEDGE:
            why["btc"].append(coin)
            continue
        if d0 > until:
            why["after_window"].append(coin)
            continue
        b = binance_symbol(coin, bset)
        if b is not None and b in listings.RENAMES:
            why["renamed_token"].append(coin)
            continue
        if b is not None and binance_first[b] < d0 - OLD_LEAD:
            why["old_token"].append(coin)
            continue
        ev[coin] = pd.Timestamp(d0)
    s = pd.Series(ev, dtype=object)
    return (s.sort_values() if len(s) else s), why


def near_misses(coins, binance: set[str]) -> dict[str, list[str]]:
    """Diagnostics only: event coins with no exact Binance match but a Binance symbol ending in the same name
    (another unit multiplier, e.g. 1000000MOG), listed for review in the report; the rule itself is unchanged."""
    out = {}
    for coin in coins:
        if binance_symbol(coin, binance) is not None:
            continue
        stem = (coin[1:] if len(coin) > 1 and coin[0] == "k" and coin[1].isupper() else coin).upper() + "USDT"
        hits = sorted(b for b in binance if b.endswith(stem) and b != stem)
        if hits:
            out[coin] = hits
    return out


def basket(close: pd.DataFrame, quote_volume: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Bool day x coin: the point-in-time top-20 alt basket (BTC excluded)."""
    b = eligible({"close": close, "quote_volume": quote_volume, "volume": volume}, BASKET_RULES)
    return b[[c for c in b.columns if c != HEDGE and bool(b[c].any())]]
