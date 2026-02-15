"""Node-level fault injection: crash and pause.

``CrashNode`` and ``PauseNode`` set a ``_crashed`` flag on the target entity.
When ``_crashed`` is True, ``Event.invoke()`` silently drops events targeting
that entity (same pattern as cancelled events).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from happysimulator.faults.fault import FaultContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrashNode:
    """Crash a node at a specific time, optionally restart later.

    Sets ``entity._crashed = True`` at crash time, causing all events
    targeting the entity to be silently dropped. If ``restart_at`` is
    provided, clears the flag at that time.

    Attributes:
        entity_name: Name of the entity to crash.
        at: Crash time in seconds.
        restart_at: Optional restart time in seconds. None = permanent crash.
    """

    entity_name: str
    at: float
    restart_at: float | None = None

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        entity = ctx.entities[self.entity_name]
        entity_name = self.entity_name
        events: list[Event] = []

        def crash(e: Event) -> None:
            entity._crashed = True  # type: ignore[attr-defined]
            logger.info("[FaultInjection] Crashed '%s' at %s", entity_name, e.time)

        events.append(
            Event.once(
                time=Instant.from_seconds(self.at),
                event_type=f"fault.crash:{entity_name}",
                fn=crash,
                daemon=True,
            )
        )

        if self.restart_at is not None:

            def restart(e: Event) -> None:
                entity._crashed = False  # type: ignore[attr-defined]
                logger.info(
                    "[FaultInjection] Restarted '%s' at %s",
                    entity_name,
                    e.time,
                )

            events.append(
                Event.once(
                    time=Instant.from_seconds(self.restart_at),
                    event_type=f"fault.restart:{entity_name}",
                    fn=restart,
                    daemon=True,
                )
            )

        return events


@dataclass(frozen=True)
class PauseNode:
    """Pause a node (freeze processing) for a time window, then resume.

    Semantically identical to CrashNode but uses ``start``/``end`` naming
    to emphasize the temporary nature of the fault.

    Attributes:
        entity_name: Name of the entity to pause.
        start: Pause start time in seconds.
        end: Resume time in seconds.
    """

    entity_name: str
    start: float
    end: float

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        entity = ctx.entities[self.entity_name]
        entity_name = self.entity_name
        events: list[Event] = []

        def pause(e: Event) -> None:
            entity._crashed = True  # type: ignore[attr-defined]
            logger.info("[FaultInjection] Paused '%s' at %s", entity_name, e.time)

        def resume(e: Event) -> None:
            entity._crashed = False  # type: ignore[attr-defined]
            logger.info("[FaultInjection] Resumed '%s' at %s", entity_name, e.time)

        events.append(
            Event.once(
                time=Instant.from_seconds(self.start),
                event_type=f"fault.pause:{entity_name}",
                fn=pause,
                daemon=True,
            )
        )
        events.append(
            Event.once(
                time=Instant.from_seconds(self.end),
                event_type=f"fault.resume:{entity_name}",
                fn=resume,
                daemon=True,
            )
        )

        return events
