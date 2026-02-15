"""SWIM-style membership protocol for cluster failure detection.

Implements a membership protocol based on the SWIM (Scalable Weakly-consistent
Infection-style Membership) approach. Nodes periodically probe peers, use
indirect pings via delegates when direct probes fail, and disseminate
membership changes via infection-style piggybacking.

Uses PhiAccrualDetector for nuanced failure detection rather than binary
timeout-based detection.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.components.consensus.phi_accrual_detector import PhiAccrualDetector

logger = logging.getLogger(__name__)


class MemberState(Enum):
    """States a member can be in."""
    ALIVE = auto()
    SUSPECT = auto()
    DEAD = auto()


@dataclass
class MemberInfo:
    """Information about a cluster member.

    Attributes:
        name: The member's identifier.
        entity: Reference to the member's Entity (for network.send).
        state: Current membership state.
        incarnation: Monotonically increasing counter to override stale suspicions.
        detector: Phi accrual detector for this member.
        state_change_time: Simulation time when state last changed.
    """
    name: str
    entity: Entity
    state: MemberState = MemberState.ALIVE
    incarnation: int = 0
    detector: PhiAccrualDetector = field(default_factory=lambda: PhiAccrualDetector(threshold=8.0))
    state_change_time: float = 0.0


@dataclass(frozen=True)
class MembershipStats:
    """Snapshot of membership protocol statistics.

    Attributes:
        alive_count: Number of ALIVE members.
        suspect_count: Number of SUSPECT members.
        dead_count: Number of DEAD members.
        probes_sent: Total direct probes sent.
        indirect_probes_sent: Total indirect probes sent.
        acks_received: Total acks received.
        updates_disseminated: Total membership updates piggybacked.
    """
    alive_count: int = 0
    suspect_count: int = 0
    dead_count: int = 0
    probes_sent: int = 0
    indirect_probes_sent: int = 0
    acks_received: int = 0
    updates_disseminated: int = 0


class MembershipProtocol(Entity):
    """SWIM membership protocol entity.

    Periodically probes peers for liveness, suspects unresponsive nodes,
    and disseminates membership updates via piggybacking on messages.

    Args:
        name: Entity identifier.
        network: Network used for communication.
        probe_interval: Seconds between probe rounds.
        suspicion_timeout: Seconds before a suspected member is declared dead.
        indirect_probe_count: Number of delegates for indirect probing.
        phi_threshold: Phi threshold for the accrual detector.
    """

    def __init__(
        self,
        name: str,
        network: Any,
        probe_interval: float = 1.0,
        suspicion_timeout: float = 5.0,
        indirect_probe_count: int = 3,
        phi_threshold: float = 8.0,
    ) -> None:
        super().__init__(name)
        self._network = network
        self._probe_interval = probe_interval
        self._suspicion_timeout = suspicion_timeout
        self._indirect_probe_count = indirect_probe_count
        self._phi_threshold = phi_threshold

        self._members: dict[str, MemberInfo] = {}
        self._incarnation: int = 0
        self._pending_updates: list[dict[str, Any]] = []
        self._probe_order: list[str] = []
        self._probe_index: int = 0
        self._pending_acks: dict[str, Event] = {}  # member_name -> timeout event

        # Stats
        self._probes_sent: int = 0
        self._indirect_probes_sent: int = 0
        self._acks_received: int = 0
        self._updates_disseminated: int = 0

    def add_member(self, entity: Entity) -> None:
        """Register a peer member."""
        if entity.name == self.name:
            return
        info = MemberInfo(
            name=entity.name,
            entity=entity,
            detector=PhiAccrualDetector(
                threshold=self._phi_threshold,
                initial_interval=self._probe_interval,
            ),
        )
        self._members[entity.name] = info
        self._probe_order.append(entity.name)

    def start(self) -> list[Event]:
        """Schedule the first probe tick."""
        random.shuffle(self._probe_order)
        return [Event(
            time=self.now + self._probe_interval,
            event_type="MembershipProbeTick",
            target=self,
            daemon=True,
        )]

    def handle_event(self, event: Event):
        handlers = {
            "MembershipProbeTick": self._handle_probe_tick,
            "MembershipPing": self._handle_ping,
            "MembershipAck": self._handle_ack,
            "MembershipIndirectPing": self._handle_indirect_ping,
            "MembershipIndirectAck": self._handle_indirect_ack,
            "MembershipSuspicionTimeout": self._handle_suspicion_timeout,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        return None

    def _handle_probe_tick(self, event: Event) -> list[Event]:
        events: list[Event] = []

        # Check phi for all members and update states
        now_s = self.now.to_seconds()
        for info in self._members.values():
            if info.state == MemberState.DEAD:
                continue
            if info.state == MemberState.ALIVE:
                if not info.detector.is_available(now_s):
                    self._suspect_member(info, now_s)

        # Pick next target to probe
        target = self._next_probe_target()
        if target is not None:
            info = self._members[target]
            # Send direct ping
            ping = self._network.send(
                source=self,
                destination=info.entity,
                event_type="MembershipPing",
                payload={
                    "from": self.name,
                    "incarnation": self._incarnation,
                    "updates": self._drain_updates(),
                },
                daemon=True,
            )
            events.append(ping)
            self._probes_sent += 1

            # Schedule ack timeout → indirect probe
            timeout = Event(
                time=self.now + self._probe_interval * 0.5,
                event_type="MembershipIndirectPing",
                target=self,
                daemon=True,
                context={"metadata": {"probe_target": target}},
            )
            # Cancel previous timeout for same target
            if target in self._pending_acks:
                self._pending_acks[target].cancel()
            self._pending_acks[target] = timeout
            events.append(timeout)

        # Schedule next probe tick
        events.append(Event(
            time=self.now + self._probe_interval,
            event_type="MembershipProbeTick",
            target=self,
            daemon=True,
        ))
        return events

    def _handle_ping(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        sender = metadata.get("from")
        updates = metadata.get("updates", [])

        # Apply piggybacked updates
        self._apply_updates(updates)

        if sender is None:
            return []

        # Record heartbeat for sender
        if sender in self._members:
            self._members[sender].detector.heartbeat(self.now.to_seconds())
            if self._members[sender].state == MemberState.SUSPECT:
                self._members[sender].state = MemberState.ALIVE

        # Send ack back
        ack = self._network.send(
            source=self,
            destination=self._members[sender].entity if sender in self._members else event.target,
            event_type="MembershipAck",
            payload={
                "from": self.name,
                "ack_for": sender,
                "incarnation": self._incarnation,
                "updates": self._drain_updates(),
            },
            daemon=True,
        )
        return [ack]

    def _handle_ack(self, event: Event) -> list[Event] | None:
        metadata = event.context.get("metadata", {})
        sender = metadata.get("from")
        updates = metadata.get("updates", [])

        self._apply_updates(updates)
        self._acks_received += 1

        if sender and sender in self._members:
            self._members[sender].detector.heartbeat(self.now.to_seconds())
            if self._members[sender].state == MemberState.SUSPECT:
                self._members[sender].state = MemberState.ALIVE

            # Cancel timeout for this member
            if sender in self._pending_acks:
                self._pending_acks[sender].cancel()
                del self._pending_acks[sender]
        return None

    def _handle_indirect_ping(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        target_name = metadata.get("probe_target")

        if target_name is None or target_name not in self._members:
            return []

        target_info = self._members[target_name]

        # If we already got an ack, skip
        if target_name not in self._pending_acks:
            return []

        # Pick random delegates (excluding self and target)
        delegates = [
            name for name in self._members
            if name != target_name
            and self._members[name].state != MemberState.DEAD
        ]
        random.shuffle(delegates)
        delegates = delegates[:self._indirect_probe_count]

        events: list[Event] = []
        for delegate_name in delegates:
            delegate = self._members[delegate_name]
            msg = self._network.send(
                source=self,
                destination=delegate.entity,
                event_type="MembershipPing",
                payload={
                    "from": self.name,
                    "indirect_for": target_name,
                    "incarnation": self._incarnation,
                    "updates": self._drain_updates(),
                },
                daemon=True,
            )
            events.append(msg)
            self._indirect_probes_sent += 1

        # Schedule suspicion timeout if still no ack
        suspicion_event = Event(
            time=self.now + self._suspicion_timeout,
            event_type="MembershipSuspicionTimeout",
            target=self,
            daemon=True,
            context={"metadata": {"suspect": target_name}},
        )
        # Replace the pending ack tracker with suspicion timeout
        if target_name in self._pending_acks:
            self._pending_acks[target_name].cancel()
        self._pending_acks[target_name] = suspicion_event
        events.append(suspicion_event)

        return events

    def _handle_indirect_ack(self, event: Event) -> list[Event] | None:
        # Treat same as regular ack
        return self._handle_ack(event)

    def _handle_suspicion_timeout(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        suspect_name = metadata.get("suspect")

        if suspect_name and suspect_name in self._members:
            info = self._members[suspect_name]
            if info.state == MemberState.SUSPECT:
                info.state = MemberState.DEAD
                info.state_change_time = self.now.to_seconds()
                self._pending_updates.append({
                    "member": suspect_name,
                    "state": "dead",
                    "incarnation": info.incarnation,
                })
                logger.debug("[%s] Member %s declared DEAD", self.name, suspect_name)
            if suspect_name in self._pending_acks:
                del self._pending_acks[suspect_name]
        return None

    def _suspect_member(self, info: MemberInfo, now_s: float) -> None:
        if info.state != MemberState.ALIVE:
            return
        info.state = MemberState.SUSPECT
        info.state_change_time = now_s
        self._pending_updates.append({
            "member": info.name,
            "state": "suspect",
            "incarnation": info.incarnation,
        })
        logger.debug("[%s] Suspecting member %s", self.name, info.name)

    def _next_probe_target(self) -> str | None:
        alive = [
            name for name in self._probe_order
            if name in self._members and self._members[name].state != MemberState.DEAD
        ]
        if not alive:
            return None
        if self._probe_index >= len(alive):
            self._probe_index = 0
            random.shuffle(alive)
            self._probe_order = alive
        target = alive[self._probe_index % len(alive)]
        self._probe_index += 1
        return target

    def _drain_updates(self) -> list[dict[str, Any]]:
        updates = list(self._pending_updates)
        self._updates_disseminated += len(updates)
        self._pending_updates.clear()
        return updates

    def _apply_updates(self, updates: list[dict[str, Any]]) -> None:
        for update in updates:
            member_name = update.get("member")
            state_str = update.get("state")
            incarnation = update.get("incarnation", 0)

            if member_name not in self._members:
                continue

            info = self._members[member_name]
            # Only apply if incarnation is >= what we know
            if incarnation < info.incarnation:
                continue

            if state_str == "suspect" and info.state == MemberState.ALIVE:
                info.state = MemberState.SUSPECT
                info.incarnation = max(info.incarnation, incarnation)
            elif state_str == "dead" and info.state != MemberState.DEAD:
                info.state = MemberState.DEAD
                info.incarnation = max(info.incarnation, incarnation)
            elif state_str == "alive" and incarnation > info.incarnation:
                info.state = MemberState.ALIVE
                info.incarnation = incarnation

    @property
    def alive_members(self) -> list[str]:
        """Names of all ALIVE members."""
        return [name for name, info in self._members.items() if info.state == MemberState.ALIVE]

    @property
    def suspected_members(self) -> list[str]:
        """Names of all SUSPECT members."""
        return [name for name, info in self._members.items() if info.state == MemberState.SUSPECT]

    @property
    def dead_members(self) -> list[str]:
        """Names of all DEAD members."""
        return [name for name, info in self._members.items() if info.state == MemberState.DEAD]

    @property
    def stats(self) -> MembershipStats:
        """Current membership statistics."""
        return MembershipStats(
            alive_count=len(self.alive_members),
            suspect_count=len(self.suspected_members),
            dead_count=len(self.dead_members),
            probes_sent=self._probes_sent,
            indirect_probes_sent=self._indirect_probes_sent,
            acks_received=self._acks_received,
            updates_disseminated=self._updates_disseminated,
        )

    def get_member_state(self, name: str) -> MemberState | None:
        """Get the current state of a member by name."""
        info = self._members.get(name)
        return info.state if info else None

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"MembershipProtocol({self.name}, "
            f"alive={stats.alive_count}, "
            f"suspect={stats.suspect_count}, "
            f"dead={stats.dead_count})"
        )
