"""Inventory buffer with (s, Q) reorder policy.

InventoryBuffer manages a stock counter. "Consume" events decrement stock.
When stock falls to or below the reorder point ``s``, a replenishment order
of size ``Q`` is placed with the supplier, arriving after ``lead_time``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)

_REPLENISH = "_InventoryReplenish"


@dataclass(frozen=True)
class InventoryStats:
    """Snapshot of inventory statistics."""

    current_stock: int = 0
    stockouts: int = 0
    reorders: int = 0
    items_consumed: int = 0
    items_replenished: int = 0

    @property
    def fill_rate(self) -> float:
        """Fraction of demand satisfied from stock (0.0-1.0)."""
        total = self.items_consumed + self.stockouts
        if total == 0:
            return 1.0
        return self.items_consumed / total


class InventoryBuffer(Entity):
    """Entity managing stock with (s, Q) reorder policy.

    When stock falls to or below ``reorder_point``, a replenishment
    order of ``order_quantity`` is placed with the ``supplier``,
    arriving after ``lead_time`` seconds.

    Consume events are forwarded to ``downstream`` if provided and
    stock is available. If stock is zero, the event is counted as a
    stockout and optionally forwarded to ``stockout_target``.

    Args:
        name: Identifier for logging.
        initial_stock: Starting inventory level.
        reorder_point: Stock level that triggers reorder (s).
        order_quantity: Amount to order on each reorder (Q).
        lead_time: Seconds for replenishment to arrive.
        supplier: Entity to receive replenishment orders (optional).
        downstream: Entity to forward satisfied demand to (optional).
        stockout_target: Entity to forward stockout events to (optional).
    """

    def __init__(
        self,
        name: str,
        initial_stock: int = 100,
        reorder_point: int = 20,
        order_quantity: int = 50,
        lead_time: float = 5.0,
        supplier: Entity | None = None,
        downstream: Entity | None = None,
        stockout_target: Entity | None = None,
    ):
        if initial_stock < 0:
            raise ValueError(f"initial_stock must be >= 0, got {initial_stock}")
        if reorder_point < 0:
            raise ValueError(f"reorder_point must be >= 0, got {reorder_point}")
        if order_quantity <= 0:
            raise ValueError(f"order_quantity must be > 0, got {order_quantity}")
        super().__init__(name)
        self._stock = initial_stock
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
        self.lead_time = lead_time
        self.supplier = supplier
        self.downstream = downstream
        self.stockout_target = stockout_target

        self._stockouts = 0
        self._reorders = 0
        self._items_consumed = 0
        self._items_replenished = 0
        self._order_pending = False

    @property
    def stock(self) -> int:
        return self._stock

    @property
    def stats(self) -> InventoryStats:
        return InventoryStats(
            current_stock=self._stock,
            stockouts=self._stockouts,
            reorders=self._reorders,
            items_consumed=self._items_consumed,
            items_replenished=self._items_replenished,
        )

    def handle_event(self, event: Event) -> list[Event]:
        if event.event_type == _REPLENISH:
            return self._handle_replenish(event)
        return self._handle_consume(event)

    def _handle_consume(self, event: Event) -> list[Event]:
        from happysimulator.core.temporal import Instant

        amount = event.context.get("quantity", 1)
        results: list[Event] = []

        if self._stock >= amount:
            self._stock -= amount
            self._items_consumed += amount

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
                self._stock,
            )
            if self.stockout_target is not None:
                results.append(
                    Event(
                        time=self.now,
                        event_type="Stockout",
                        target=self.stockout_target,
                        context=event.context,
                    )
                )

        # Check reorder
        if self._stock <= self.reorder_point and not self._order_pending:
            self._order_pending = True
            self._reorders += 1
            now_s = self.now.to_seconds()
            results.append(
                Event(
                    time=Instant.from_seconds(now_s + self.lead_time),
                    event_type=_REPLENISH,
                    target=self,
                    context={"quantity": self.order_quantity},
                )
            )
            logger.debug(
                "[%s] Reorder #%d placed: Q=%d, arriving at t=%.2f",
                self.name,
                self._reorders,
                self.order_quantity,
                now_s + self.lead_time,
            )

        return results

    def _handle_replenish(self, event: Event) -> list[Event]:
        quantity = event.context.get("quantity", self.order_quantity)
        self._stock += quantity
        self._items_replenished += quantity
        self._order_pending = False
        logger.debug(
            "[%s] Replenished %d units, stock now %d",
            self.name,
            quantity,
            self._stock,
        )
        return []
