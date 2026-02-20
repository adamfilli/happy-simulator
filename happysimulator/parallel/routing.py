"""Event router factory for parallel simulation partitions.

The router separates events produced by ``Entity.handle_event()`` into
*local* events (pushed to the partition's own heap) and *cross-partition*
events (appended to an outbox for delivery during the barrier phase).
"""

from __future__ import annotations

from collections.abc import Callable

from happysimulator.core.callback_entity import CallbackEntity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


def make_event_router(
    partition_name: str,
    local_entity_ids: frozenset[int],
    linked_entity_ids: frozenset[int],
    outbox: list[tuple[Event, Instant]],
) -> Callable[[list[Event], Instant], list[Event]]:
    """Create a routing closure for a partition.

    Args:
        partition_name: Name of this partition (for error messages).
        local_entity_ids: ``id()`` values of entities in this partition.
        linked_entity_ids: ``id()`` values of entities reachable via links
            from this partition.
        outbox: Mutable list where ``(event, send_time)`` tuples are appended
            for cross-partition events.

    Returns:
        A callable ``(events, current_time) -> local_events``.

    Raises:
        RuntimeError: If an event targets an entity in an unlinked partition.
    """

    def route(events: list[Event], current_time: Instant) -> list[Event]:
        local: list[Event] = []
        for event in events:
            target = event.target
            # CallbackEntity targets are always local (e.g. Event.once())
            if isinstance(target, CallbackEntity):
                local.append(event)
                continue

            tid = id(target)
            if tid in local_entity_ids:
                local.append(event)
            elif tid in linked_entity_ids:
                outbox.append((event, current_time))
            else:
                target_name = getattr(target, "name", repr(target))
                raise RuntimeError(
                    f"Partition '{partition_name}': event targets entity "
                    f"'{target_name}' which is not in this partition and "
                    f"not reachable via any PartitionLink"
                )
        return local

    return route
