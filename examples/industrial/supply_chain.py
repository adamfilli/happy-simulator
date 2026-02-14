"""Multi-echelon supply chain simulation demonstrating the bullwhip effect.

Three-tier supply chain: Factory -> Distributor -> Retailer. Each tier uses
an InventoryBuffer with (s, Q) reorder policy. Stochastic retail demand via
Poisson arrivals drives the chain. Each tier's fulfilled orders become
consumption at the next tier downstream (customer). Upstream propagation
happens when a DemandAmplifier entity detects that a tier has reordered,
and forwards a batch Consume event to the upstream tier, amplifying the
order size and creating the classic bullwhip effect.

## Architecture Diagram

```
                     MULTI-ECHELON SUPPLY CHAIN
    +---------------------------------------------------------------+
    |                                                               |
    |  Source ------> DemandTap -----> Retailer -----> CustomerSink |
    | (Poisson)      (tracks demand)   (s=50, Q=100)               |
    |                                  lead_time=2hr               |
    |                                                               |
    |  DemandAmplifier[Retail] --amplify--> Distributor             |
    |    (polls retailer reorders,          (s=150, Q=300)          |
    |     sends batch Consume upstream)     lead_time=5hr          |
    |                                                               |
    |  DemandAmplifier[Dist] --amplify--> Factory                   |
    |    (polls distributor reorders,       (s=500, Q=1000)         |
    |     sends batch Consume upstream)     lead_time=10hr         |
    |                                                               |
    |  Stockout counters at each tier                               |
    |  Variance amplification measured via DemandTap entities       |
    +---------------------------------------------------------------+
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from happysimulator import (
    Entity,
    Event,
    Instant,
    LatencyTracker,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.common import Counter
from happysimulator.components.industrial import InventoryBuffer


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class SupplyChainConfig:
    duration_s: float = 86400.0     # 24 hours (1 day)
    demand_rate: float = 0.008      # orders per second (~29/hr)
    # Retailer
    retailer_stock: int = 100
    retailer_reorder_point: int = 30
    retailer_order_qty: int = 50
    retailer_lead_time: float = 7200.0   # 2 hours
    # Distributor
    distributor_stock: int = 300
    distributor_reorder_point: int = 100
    distributor_order_qty: int = 150
    distributor_lead_time: float = 18000.0  # 5 hours
    # Factory
    factory_stock: int = 1000
    factory_reorder_point: int = 300
    factory_order_qty: int = 500
    factory_lead_time: float = 36000.0  # 10 hours (production)
    seed: int = 42


# =============================================================================
# Demand Tap: passes events through and records timestamps for analysis
# =============================================================================

class DemandTap(Entity):
    """Transparent tap that records demand events and forwards to downstream."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self.downstream = downstream
        self.event_times: list[float] = []
        self.event_quantities: list[int] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.event_times.append(self.now.to_seconds())
        self.event_quantities.append(event.context.get("quantity", 1))
        return [
            Event(time=self.now, event_type=event.event_type,
                  target=self.downstream, context=event.context)
        ]

    @property
    def total_events(self) -> int:
        return len(self.event_times)

    @property
    def total_quantity(self) -> int:
        return sum(self.event_quantities)

    def quantity_variance(self, window_s: float = 3600.0) -> float:
        """Compute variance of demand quantity across time windows."""
        if len(self.event_times) < 2:
            return 0.0
        max_t = max(self.event_times)
        num_windows = max(1, int(max_t / window_s))
        quantities = [0] * num_windows
        for t, q in zip(self.event_times, self.event_quantities):
            idx = min(int(t / window_s), num_windows - 1)
            quantities[idx] += q
        if len(quantities) < 2:
            return 0.0
        mean = sum(quantities) / len(quantities)
        return sum((q - mean) ** 2 for q in quantities) / len(quantities)

    def quantity_mean(self, window_s: float = 3600.0) -> float:
        """Compute mean demand quantity per window."""
        if not self.event_times:
            return 0.0
        max_t = max(self.event_times)
        num_windows = max(1, int(max_t / window_s))
        return sum(self.event_quantities) / num_windows


# =============================================================================
# Demand Amplifier: monitors a tier's reorders and creates upstream demand
# =============================================================================

class DemandAmplifier(Entity):
    """Periodically checks an inventory tier and sends batch consume upstream.

    Every check_interval, it looks at whether the watched tier has reordered
    since last check. If so, it sends an amplified consume event to the
    upstream tier, modeling the batch-ordering effect that drives bullwhip.
    """

    _CHECK = "_DemandAmpCheck"

    def __init__(self, name: str, watched: InventoryBuffer,
                 upstream_tap: DemandTap, check_interval: float = 300.0):
        super().__init__(name)
        self.watched = watched
        self.upstream_tap = upstream_tap
        self.check_interval = check_interval
        self._last_reorder_count = 0
        self.amplified_orders = 0

    def start_event(self) -> Event:
        """Create the initial check event."""
        return Event(
            time=Instant.from_seconds(self.check_interval),
            event_type=self._CHECK,
            target=self,
        )

    def handle_event(self, event: Event) -> list[Event]:
        results: list[Event] = []

        current_reorders = self.watched.stats.reorders
        new_reorders = current_reorders - self._last_reorder_count

        if new_reorders > 0:
            # Each reorder at this tier creates amplified demand upstream
            order_qty = self.watched.order_quantity
            for _ in range(new_reorders):
                self.amplified_orders += 1
                results.append(
                    Event(
                        time=self.now,
                        event_type="Consume",
                        target=self.upstream_tap,
                        context={
                            "quantity": order_qty,
                            "created_at": self.now,
                            "source": self.name,
                        },
                    )
                )

        self._last_reorder_count = current_reorders

        # Schedule next check
        now_s = self.now.to_seconds()
        results.append(
            Event(
                time=Instant.from_seconds(now_s + self.check_interval),
                event_type=self._CHECK,
                target=self,
            )
        )

        return results


# =============================================================================
# Main Simulation
# =============================================================================

@dataclass
class SupplyChainResult:
    customer_sink: LatencyTracker
    stockout_counters: dict[str, Counter]
    inventories: dict[str, InventoryBuffer]
    demand_taps: dict[str, DemandTap]
    amplifiers: dict[str, DemandAmplifier]
    config: SupplyChainConfig
    summary: SimulationSummary


def run_supply_chain_simulation(config: SupplyChainConfig | None = None) -> SupplyChainResult:
    if config is None:
        config = SupplyChainConfig()
    random.seed(config.seed)

    customer_sink = LatencyTracker("CustomerSink")

    # Stockout counters
    retail_stockout = Counter("RetailStockout")
    dist_stockout = Counter("DistStockout")
    factory_stockout = Counter("FactoryStockout")

    # Factory (top tier)
    factory = InventoryBuffer(
        "Factory",
        initial_stock=config.factory_stock,
        reorder_point=config.factory_reorder_point,
        order_quantity=config.factory_order_qty,
        lead_time=config.factory_lead_time,
        stockout_target=factory_stockout,
    )

    # Tap for factory demand
    factory_tap = DemandTap("FactoryTap", downstream=factory)

    # Distributor
    distributor = InventoryBuffer(
        "Distributor",
        initial_stock=config.distributor_stock,
        reorder_point=config.distributor_reorder_point,
        order_quantity=config.distributor_order_qty,
        lead_time=config.distributor_lead_time,
        stockout_target=dist_stockout,
    )

    # Tap for distributor demand
    dist_tap = DemandTap("DistTap", downstream=distributor)

    # Retailer
    retailer = InventoryBuffer(
        "Retailer",
        initial_stock=config.retailer_stock,
        reorder_point=config.retailer_reorder_point,
        order_quantity=config.retailer_order_qty,
        lead_time=config.retailer_lead_time,
        downstream=customer_sink,
        stockout_target=retail_stockout,
    )

    # Tap for retail demand (from customer source)
    retail_tap = DemandTap("RetailTap", downstream=retailer)

    # Amplifiers: link tiers upstream
    retail_amplifier = DemandAmplifier(
        "RetailAmplifier", watched=retailer,
        upstream_tap=dist_tap, check_interval=300.0,
    )
    dist_amplifier = DemandAmplifier(
        "DistAmplifier", watched=distributor,
        upstream_tap=factory_tap, check_interval=300.0,
    )

    # Customer demand source (flows through retail tap)
    source = Source.poisson(
        rate=config.demand_rate, target=retail_tap,
        event_type="Consume", name="CustomerDemand",
        stop_after=config.duration_s,
    )

    entities = [
        retailer, distributor, factory,
        retail_tap, dist_tap, factory_tap,
        retail_amplifier, dist_amplifier,
        customer_sink,
        retail_stockout, dist_stockout, factory_stockout,
    ]

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(config.duration_s + 43200),  # 12hr drain
        sources=[source],
        entities=entities,
    )

    # Start amplifier check loops
    sim.schedule(retail_amplifier.start_event())
    sim.schedule(dist_amplifier.start_event())

    summary = sim.run()

    return SupplyChainResult(
        customer_sink=customer_sink,
        stockout_counters={
            "Retailer": retail_stockout,
            "Distributor": dist_stockout,
            "Factory": factory_stockout,
        },
        inventories={
            "Retailer": retailer,
            "Distributor": distributor,
            "Factory": factory,
        },
        demand_taps={
            "Retailer": retail_tap,
            "Distributor": dist_tap,
            "Factory": factory_tap,
        },
        amplifiers={
            "RetailAmplifier": retail_amplifier,
            "DistAmplifier": dist_amplifier,
        },
        config=config,
        summary=summary,
    )


def print_summary(result: SupplyChainResult) -> None:
    print("\n" + "=" * 65)
    print("MULTI-ECHELON SUPPLY CHAIN SIMULATION RESULTS")
    print("=" * 65)

    c = result.config
    print(f"\nConfiguration:")
    print(f"  Duration: {c.duration_s/3600:.0f} hours")
    print(f"  Customer demand rate: {c.demand_rate*3600:.0f} orders/hr")

    initial_stocks = {
        "Retailer": c.retailer_stock,
        "Distributor": c.distributor_stock,
        "Factory": c.factory_stock,
    }

    print(f"\nInventory Levels:")
    for name, inv in result.inventories.items():
        stats = inv.stats
        print(f"  {name}:")
        print(f"    Initial: {initial_stocks[name]}, Final: {stats.current_stock}")
        print(f"    Consumed: {stats.items_consumed}, Replenished: {stats.items_replenished}")
        print(f"    Reorders: {stats.reorders}")
        print(f"    Fill rate: {stats.fill_rate*100:.1f}%")

    print(f"\nStockouts:")
    for name, counter in result.stockout_counters.items():
        print(f"  {name}: {counter.total}")

    print(f"\nDemand Amplification:")
    for name, amp in result.amplifiers.items():
        print(f"  {name}: {amp.amplified_orders} batch orders sent upstream")

    # Bullwhip analysis
    print(f"\nBullwhip Effect (demand quantity per 1-hr window):")
    window = 3600.0
    variances = {}
    for name, tap in result.demand_taps.items():
        m = tap.quantity_mean(window)
        v = tap.quantity_variance(window)
        variances[name] = v
        print(f"  {name}: {tap.total_quantity} total units, "
              f"mean={m:.1f} units/hr, variance={v:.1f}")

    retail_var = variances.get("Retailer", 0)
    if retail_var > 0:
        print(f"\n  Variance amplification (vs Retailer):")
        for name, v in variances.items():
            if name != "Retailer":
                ratio = v / retail_var
                print(f"    {name}: {ratio:.2f}x")

    print(f"\nCustomer Fulfillment:")
    print(f"  Orders fulfilled: {result.customer_sink.count}")
    total_demand = result.demand_taps["Retailer"].total_events
    if total_demand > 0:
        fill_pct = result.customer_sink.count / total_demand * 100
        print(f"  Overall fill rate: {fill_pct:.1f}%")

    print(f"\n{result.summary}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-echelon supply chain simulation")
    parser.add_argument("--duration", type=float, default=86400.0,
                        help="Duration in seconds (default: 86400 = 24hr)")
    parser.add_argument("--demand-rate", type=float, default=0.008,
                        help="Demand rate per second (default: 0.008)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    config = SupplyChainConfig(
        duration_s=args.duration,
        demand_rate=args.demand_rate,
        seed=args.seed,
    )
    result = run_supply_chain_simulation(config)
    print_summary(result)
