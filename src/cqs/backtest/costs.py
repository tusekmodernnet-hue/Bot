"""Trading cost model. Every backtest must use one; there is no cost-free mode.

Defaults are conservative assumptions from the Stage 1 protocol, to be confirmed against the execution venue:
- taker fee 0.05% per side, no maker rebates assumed;
- slippage 2 bp for BTC/ETH, more for less liquid contracts (pass per-asset values);
- forced exit on delisting or missing data at the last close with 2% extra slippage.
Funding is charged from actual archive rates by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0005
    default_slippage_bps: float = 2.0
    slippage_bps: Mapping[str, float] = field(default_factory=dict)
    forced_exit_slippage_bps: float = 200.0
    charge_funding: bool = True

    def slippage(self, asset: str) -> float:
        return self.slippage_bps.get(asset, self.default_slippage_bps) / 1e4

    def rate(self, asset: str) -> float:
        """Cost per unit of traded notional (fee + slippage)."""
        return self.taker_fee + self.slippage(asset)

    def forced_rate(self) -> float:
        return self.taker_fee + self.forced_exit_slippage_bps / 1e4

    def scaled(self, k: float) -> "CostModel":
        """Stress test: multiply fees and slippage by k (the protocol requires k=2 to stay profitable)."""
        return replace(
            self,
            taker_fee=self.taker_fee * k,
            default_slippage_bps=self.default_slippage_bps * k,
            slippage_bps={a: v * k for a, v in self.slippage_bps.items()},
            forced_exit_slippage_bps=self.forced_exit_slippage_bps * k,
        )
