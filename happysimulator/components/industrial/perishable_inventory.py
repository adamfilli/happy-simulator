"""Inventory with shelf-life expiration and spoilage sweeps.

PerishableInventory extends the inventory concept with item expiration.
Items are stored as FIFO batches with arrival timestamps. Periodic
spoilage checks remove expired items, and waste statistics are tracked.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)

_SPOILAGE_CHECK = "_SpoilageCheck"
_REPLENISH = "_PerishableReplenish"


@dataclass(frozen=True)
class PerishableInventoryStats:
    """Snapshot of perishable inventory statistics."""

    current_stock: int = 0
    total_consumed: int = 0
    total_spoiled: int = 0
    stockouts: int = 0
    reorders: int = 0

    @property
    def waste_rate(self) -> float:
        """Fraction of items spoiled vs total received (0.0-1.0)."""
        total = self.total_consumed + self.total_spoiled
        if total == 0:
            return 0.0
        return self.total_spoiled / total


class PerishableInventory(Entity):
    """Entity managing perishable stock with periodic spoilage checks.

    Items are stored as ``(arrival_time, quantity)`` batches in a FIFO
    deque. A self-perpetuating spoilage check event periodically sweeps
    expired batches. The ``(s, Q)`` reorder policy triggers restocking
    when stock drops below ``reorder_point``.

    Args:
        name: Identifier for logging.
        initial_stock: Starting inventory level.
        shelf_life_s: Seconds before items expire.
        spoilage_check_interval_s: Seconds between spoilage sweeps.
        reorder_point: Stock level that triggers reorder.
        order_quantity: Amount to order on each reorder.
        lead_time: Seconds for replenishment to arrive.
        downstream: Entity to forward fulfilled demand to (optional).
        waste_target: Entity to notify of spoilage events (optional).
    """

    def __init__(
        self,
        name: str,
        initial_stock: int = 100,
        shelf_life_s: float = 3600.0,
        spoilage_check_interval_s: float = 60.0,
        reorder_point: int = 20,
        order_quantity: int = 50,
        lead_time: float = 5.0,
        downstream: Entity | None = None,
        waste_target: Entity | None = None,
        initial_stock_time: float | None = None,
    ):
        super().__init__(name)
        self.shelf_life_s = shelf_life_s
        self.spoilage_check_interval_s = spoilage_check_interval_s
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
        self.lead_time = lead_time
        self.downstream = downstream
        self.waste_target = waste_target

        # FIFO batches: (arrival_instant, quantity)
        from happysimulator.core.temporal import Instant

        self._items: deque[tuple[Instant, int]] = deque()
        self._deferred_initial_stock = 0
        if initial_stock > 0:
            if initial_stock_time is not None:
                self._items.append((Instant.from_seconds(initial_stock_time), initial_stock))
            else:
                # Defer initialization until first event so items use simulation start time
                self._deferred_initial_stock = initial_stock

        self._total_consumed = 0
        self._total_spoiled = 0
        self._stockouts = 0
        self._reorders = 0
        self._order_pending = False

    @property
    def stock(self) -> int:
        return self._deferred_initial_stock + sum(qty for _, qty in self._items)

    @property
    def stats(self) -> PerishableInventoryStats:
        return PerishableInventoryStats(
            current_stock=self.stock,
            total_consumed=self._total_consumed,
            total_spoiled=self._total_spoiled,
            stockouts=self._stockouts,
            reorders=self._reorders,
        )

    def start_event(self) -> Event:
        """Create the initial spoilage check event."""
        from happysimulator.core.temporal import Instant

        return Event(
            time=Instant.from_seconds(self.spoilage_check_interval_s),
            event_type=_SPOILAGE_CHECK,
            target=self,
            daemon=True,
        )

    def handle_event(self, event: Event) -> list[Event]:
        if self._deferred_initial_stock > 0:
            self._items.append((self.now, self._deferred_initial_stock))
            self._deferred_initial_stock = 0
        if event.event_type == _SPOILAGE_CHECK:
            return self._handle_spoilage_check()
        if event.event_type == _REPLENISH:
            return self._handle_replenish(event)
        return self._handle_consume(event)

    def _handle_spoilage_check(self) -> list[Event]:
        from happysimulator.core.temporal import Instant

        now = self.now
        spoiled = 0

        while self._items:
            arrival, qty = self._items[0]
            age = (now - arrival).to_seconds()
            if age >= self.shelf_life_s:
                self._items.popleft()
                spoiled += qty
            else:
                break

        if spoiled > 0:
            self._total_spoiled += spoiled
            logger.debug(
                "[%s] Spoilage check: %d items expired, stock now %d",
                self.name,
                spoiled,
                self.stock,
            )

        results: list[Event] = []

        if spoiled > 0 and self.waste_target is not None:
            results.append(
                Event(
                    time=self.now,
                    event_type="Spoiled",
                    target=self.waste_target,
                    context={"quantity": spoiled},
                )
            )

        # Check reorder after spoilage
        results.extend(self._check_reorder())

        # Schedule next spoilage check
        now_s = now.to_seconds()
        results.append(
            Event(
                time=Instant.from_seconds(now_s + self.spoilage_check_interval_s),
                event_type=_SPOILAGE_CHECK,
                target=self,
                daemon=True,
            )
        )

        return results

    def _handle_consume(self, event: Event) -> list[Event]:
        amount = event.context.get("quantity", 1)
        current = self.stock
        results: list[Event] = []

        if current >= amount:
            self._consume_fifo(amount)
            self._total_consumed += amount

            if self.downstream is not None:
                results.append(
                    Event(
                        time=self.now,
                        event_type="Fulfilled",
                        target=self.downstream,
                        context=event.context,
                    )
                )
        else:
            self._stockouts += 1
            logger.debug(
                "[%s] Stockout: requested %d, have %d",
                self.name,
                amount,
                current,
            )

        results.extend(self._check_reorder())
        return results

    def _consume_fifo(self, amount: int) -> None:
        """Remove ``amount`` items from oldest batches first."""
        remaining = amount
        while remaining > 0 and self._items:
            arrival, qty = self._items[0]
            if qty <= remaining:
                self._items.popleft()
                remaining -= qty
            else:
                self._items[0] = (arrival, qty - remaining)
                remaining = 0

    def _handle_replenish(self, event: Event) -> list[Event]:
        quantity = event.context.get("quantity", self.order_quantity)
        self._items.append((self.now, quantity))
        self._order_pending = False
        logger.debug(
            "[%s] Replenished %d units, stock now %d",
            self.name,
            quantity,
            self.stock,
        )
        return []

    def _check_reorder(self) -> list[Event]:
        from happysimulator.core.temporal import Instant

        if self.stock <= self.reorder_point and not self._order_pending:
            self._order_pending = True
            self._reorders += 1
            now_s = self.now.to_seconds()
            logger.debug(
                "[%s] Reorder #%d placed: Q=%d, arriving at t=%.2f",
                self.name,
                self._reorders,
                self.order_quantity,
                now_s + self.lead_time,
            )
            return [
                Event(
                    time=Instant.from_seconds(now_s + self.lead_time),
                    event_type=_REPLENISH,
                    target=self,
                    context={"quantity": self.order_quantity},
                )
            ]
        return []
