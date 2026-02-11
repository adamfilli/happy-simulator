"""Leader election entity using pluggable election strategies.

Monitors membership and triggers elections when the current leader fails.
Tracks the current term and leader, and notifies the cluster of leadership
changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.components.consensus.election_strategies import ElectionStrategy, BullyStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElectionStats:
    """Statistics snapshot from a LeaderElection entity.

    Attributes:
        current_leader: Name of the current leader, or None.
        current_term: Current election term.
        elections_started: Total elections initiated.
        elections_won: Elections won by this node.
        elections_participated: Elections this node participated in.
    """
    current_leader: str | None
    current_term: int
    elections_started: int
    elections_won: int
    elections_participated: int


class LeaderElection(Entity):
    """Leader election entity with pluggable strategy.

    Monitors a MembershipProtocol for leader failure and triggers elections
    using the configured ElectionStrategy.

    Args:
        name: Entity identifier.
        network: Network for communication.
        members: Dict mapping member names to Entity references.
        strategy: Election strategy to use. Defaults to BullyStrategy.
        election_timeout: Seconds to wait before starting an election.
        heartbeat_interval: Seconds between leader heartbeats.
    """

    def __init__(
        self,
        name: str,
        network: Any,
        members: dict[str, Entity] | None = None,
        strategy: ElectionStrategy | None = None,
        election_timeout: float = 2.0,
        heartbeat_interval: float = 0.5,
    ) -> None:
        super().__init__(name)
        self._network = network
        self._members: dict[str, Entity] = dict(members) if members else {}
        self._strategy = strategy or BullyStrategy()
        self._election_timeout = election_timeout
        self._heartbeat_interval = heartbeat_interval

        self._current_leader: str | None = None
        self._current_term: int = 0
        self._election_in_progress: bool = False
        self._last_leader_heartbeat: float = 0.0
        self._timeout_event: Event | None = None

        # Stats
        self._elections_started: int = 0
        self._elections_won: int = 0
        self._elections_participated: int = 0

    def add_member(self, entity: Entity) -> None:
        """Register a member for election participation."""
        self._members[entity.name] = entity

    @property
    def current_leader(self) -> str | None:
        """The current leader's name, or None."""
        return self._current_leader

    @property
    def current_term(self) -> int:
        """The current election term."""
        return self._current_term

    @property
    def is_leader(self) -> bool:
        """Whether this node is the current leader."""
        return self._current_leader == self.name

    def start(self) -> list[Event]:
        """Schedule the first election timeout check."""
        self._last_leader_heartbeat = self.now.to_seconds() if self._clock else 0.0
        timeout = Event(
            time=self.now + self._election_timeout,
            event_type="ElectionTimeoutCheck",
            target=self,
            daemon=True,
        )
        self._timeout_event = timeout
        return [timeout]

    def handle_event(self, event: Event):
        handlers = {
            "ElectionTimeoutCheck": self._handle_timeout_check,
            "ElectionChallenge": self._handle_election_message,
            "ElectionSuppress": self._handle_election_message,
            "ElectionVictory": self._handle_election_message,
            "ElectionToken": self._handle_election_message,
            "ElectionBallot": self._handle_election_message,
            "ElectionBallotResponse": self._handle_election_message,
            "LeaderHeartbeat": self._handle_leader_heartbeat,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        return None

    def _handle_timeout_check(self, event: Event) -> list[Event]:
        events: list[Event] = []
        now_s = self.now.to_seconds()

        if self.is_leader:
            # Leader sends heartbeats
            for member_name, member in self._members.items():
                if member_name != self.name:
                    hb = self._network.send(
                        source=self,
                        destination=member,
                        event_type="LeaderHeartbeat",
                        payload={
                            "leader": self.name,
                            "term": self._current_term,
                        },
                        daemon=True,
                    )
                    events.append(hb)
        elif not self._election_in_progress:
            # Check if leader timed out
            if now_s - self._last_leader_heartbeat > self._election_timeout:
                events.extend(self._start_election())

        # Schedule next check
        interval = self._heartbeat_interval if self.is_leader else self._election_timeout
        timeout = Event(
            time=self.now + interval,
            event_type="ElectionTimeoutCheck",
            target=self,
            daemon=True,
        )
        self._timeout_event = timeout
        events.append(timeout)
        return events

    def _handle_leader_heartbeat(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        leader = metadata.get("leader")
        term = metadata.get("term", 0)

        if term >= self._current_term:
            self._current_leader = leader
            self._current_term = term
            self._last_leader_heartbeat = self.now.to_seconds()
            self._election_in_progress = False
        return None

    def _handle_election_message(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        self._elections_participated += 1

        alive = list(self._members.keys())
        result = self._strategy.handle_election_message(
            node_id=self.name,
            message_type=event.event_type,
            payload=metadata,
            alive_members=alive,
        )

        events: list[Event] = []

        # Send response messages
        for msg in result.get("response_messages", []):
            target_name = msg["target"]
            if target_name in self._members:
                evt = self._network.send(
                    source=self,
                    destination=self._members[target_name],
                    event_type=msg["event_type"],
                    payload=msg["payload"],
                    daemon=True,
                )
                events.append(evt)

        # Update leader if decided
        leader = result.get("leader")
        if leader is not None:
            self._current_leader = leader
            self._current_term += 1
            self._last_leader_heartbeat = self.now.to_seconds()
            self._election_in_progress = False
            if leader == self.name:
                self._elections_won += 1

        # Start own election if strategy says so (Bully pattern)
        if result.get("start_own_election") and not self._election_in_progress:
            events.extend(self._start_election())

        if result.get("suppress_election"):
            self._election_in_progress = False

        return events

    def _start_election(self) -> list[Event]:
        self._election_in_progress = True
        self._elections_started += 1
        self._current_term += 1

        alive = list(self._members.keys())
        messages = self._strategy.get_election_messages(
            node_id=self.name,
            alive_members=alive,
            term=self._current_term,
        )

        events: list[Event] = []
        for msg in messages:
            target_name = msg["target"]
            if target_name in self._members:
                evt = self._network.send(
                    source=self,
                    destination=self._members[target_name],
                    event_type=msg["event_type"],
                    payload=msg["payload"],
                    daemon=True,
                )
                events.append(evt)

        # If no messages were generated (no higher nodes in Bully),
        # we become leader directly
        if not messages:
            self._current_leader = self.name
            self._elections_won += 1
            self._election_in_progress = False

        # Check if all messages are victory announcements
        if messages and all(m["event_type"] == "ElectionVictory" for m in messages):
            self._current_leader = self.name
            self._elections_won += 1
            self._election_in_progress = False

        return events

    @property
    def stats(self) -> ElectionStats:
        return ElectionStats(
            current_leader=self._current_leader,
            current_term=self._current_term,
            elections_started=self._elections_started,
            elections_won=self._elections_won,
            elections_participated=self._elections_participated,
        )

    def __repr__(self) -> str:
        return (
            f"LeaderElection({self.name}, "
            f"leader={self._current_leader}, "
            f"term={self._current_term})"
        )
