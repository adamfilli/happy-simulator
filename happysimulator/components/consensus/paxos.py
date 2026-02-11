"""Single-decree Paxos consensus protocol.

Implements the classic Paxos algorithm for reaching agreement on a single
value among a group of nodes. Each PaxosNode acts as both proposer and
acceptor.

The protocol proceeds in two phases:
- Phase 1 (Prepare/Promise): Proposer sends Prepare with ballot, acceptors
  respond with Promise (including any previously accepted value).
- Phase 2 (Accept/Accepted): Proposer sends Accept with chosen value,
  acceptors respond with Accepted.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class Ballot:
    """Totally ordered ballot number for Paxos proposals.

    Ballots are ordered first by number, then by node_id for tie-breaking.

    Attributes:
        number: The ballot sequence number.
        node_id: The proposer's identifier for tie-breaking.
    """
    number: int
    node_id: str


@dataclass(frozen=True)
class PaxosStats:
    """Statistics snapshot from a PaxosNode.

    Attributes:
        proposals_started: Number of proposals initiated.
        proposals_succeeded: Number of successfully decided proposals.
        proposals_failed: Number of proposals that failed (nacked).
        promises_received: Number of promise responses received.
        nacks_received: Number of nack responses received.
        accepts_received: Number of accepted responses received.
        decided_value: The decided value, or None if not yet decided.
    """
    proposals_started: int
    proposals_succeeded: int
    proposals_failed: int
    promises_received: int
    nacks_received: int
    accepts_received: int
    decided_value: Any


class PaxosNode(Entity):
    """Single-decree Paxos participant (proposer + acceptor).

    Args:
        name: Node identifier.
        network: Network for inter-node communication.
        peers: List of peer PaxosNode entities.
        retry_delay: Base delay before retrying with higher ballot.
    """

    def __init__(
        self,
        name: str,
        network: Any,
        peers: list[PaxosNode] | None = None,
        retry_delay: float = 0.5,
    ) -> None:
        super().__init__(name)
        self._network = network
        self._peers: list[PaxosNode] = list(peers) if peers else []
        self._retry_delay = retry_delay

        # Acceptor state
        self._promised_ballot: Ballot | None = None
        self._accepted_ballot: Ballot | None = None
        self._accepted_value: Any = None

        # Proposer state
        self._current_ballot: Ballot = Ballot(0, self.name)
        self._proposal_futures: dict[int, SimFuture] = {}  # ballot_number -> future
        self._phase1_responses: dict[int, list[dict]] = {}  # ballot -> [responses]
        self._phase2_responses: dict[int, int] = {}  # ballot -> count
        self._proposed_values: dict[int, Any] = {}  # ballot -> value

        # Decision
        self._decided: bool = False
        self._decided_value: Any = None

        # Stats
        self._proposals_started: int = 0
        self._proposals_succeeded: int = 0
        self._proposals_failed: int = 0
        self._promises_received: int = 0
        self._nacks_received: int = 0
        self._accepts_received: int = 0

    def set_peers(self, peers: list[PaxosNode]) -> None:
        """Set the peer list (excluding self)."""
        self._peers = [p for p in peers if p.name != self.name]

    @property
    def quorum_size(self) -> int:
        """Majority quorum: (total_nodes // 2) + 1."""
        total = len(self._peers) + 1  # peers + self
        return (total // 2) + 1

    @property
    def is_decided(self) -> bool:
        """Whether this node has learned a decided value."""
        return self._decided

    @property
    def decided_value(self) -> Any:
        """The decided value, or None if not yet decided."""
        return self._decided_value

    def propose(self, value: Any) -> SimFuture:
        """Start a proposal for a value.

        Returns a SimFuture that resolves with the decided value when
        consensus is reached.

        Args:
            value: The value to propose.

        Returns:
            SimFuture that resolves with the decided value.
        """
        if self._decided:
            future = SimFuture()
            future.resolve(self._decided_value)
            return future

        future = SimFuture()
        self._proposals_started += 1

        # Generate new ballot higher than anything we've seen
        max_seen = self._current_ballot.number
        if self._promised_ballot:
            max_seen = max(max_seen, self._promised_ballot.number)
        new_number = max_seen + 1
        self._current_ballot = Ballot(new_number, self.name)

        self._proposal_futures[new_number] = future
        self._proposed_values[new_number] = value
        self._phase1_responses[new_number] = []
        self._phase2_responses[new_number] = 0

        return future

    def start_phase1(self) -> list[Event]:
        """Send Prepare messages for the current ballot.

        Call this after propose() to begin Phase 1.
        """
        ballot = self._current_ballot
        events: list[Event] = []

        for peer in self._peers:
            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="PaxosPrepare",
                payload={
                    "ballot_number": ballot.number,
                    "ballot_node": ballot.node_id,
                },
                daemon=True,
            )
            events.append(msg)

        # Self-promise
        self._handle_prepare_internal(ballot)

        return events

    def handle_event(self, event: Event):
        handlers = {
            "PaxosPrepare": self._handle_prepare,
            "PaxosPromise": self._handle_promise,
            "PaxosNack": self._handle_nack,
            "PaxosAccept": self._handle_accept,
            "PaxosAccepted": self._handle_accepted,
            "PaxosDecided": self._handle_decided,
            "PaxosRetry": self._handle_retry,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        return None

    def _handle_prepare_internal(self, ballot: Ballot) -> None:
        """Self-promise (proposer is also acceptor)."""
        if self._promised_ballot is None or ballot >= self._promised_ballot:
            self._promised_ballot = ballot
            response = {
                "from": self.name,
                "accepted_ballot": (self._accepted_ballot.number, self._accepted_ballot.node_id) if self._accepted_ballot else None,
                "accepted_value": self._accepted_value,
            }
            # Add to our own phase1 responses
            if ballot.number in self._phase1_responses:
                self._phase1_responses[ballot.number].append(response)
                self._promises_received += 1

    def _handle_prepare(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot = Ballot(metadata["ballot_number"], metadata["ballot_node"])
        sender_name = metadata.get("source")

        # Find sender entity
        sender = self._find_peer(metadata.get("source"))
        if sender is None:
            return []

        if self._promised_ballot is not None and ballot < self._promised_ballot:
            # Nack: we've already promised a higher ballot
            nack = self._network.send(
                source=self,
                destination=sender,
                event_type="PaxosNack",
                payload={
                    "ballot_number": ballot.number,
                    "ballot_node": ballot.node_id,
                    "highest_ballot_number": self._promised_ballot.number,
                    "highest_ballot_node": self._promised_ballot.node_id,
                },
                daemon=True,
            )
            return [nack]

        # Promise
        self._promised_ballot = ballot
        promise = self._network.send(
            source=self,
            destination=sender,
            event_type="PaxosPromise",
            payload={
                "ballot_number": ballot.number,
                "ballot_node": ballot.node_id,
                "from": self.name,
                "accepted_ballot_number": self._accepted_ballot.number if self._accepted_ballot else None,
                "accepted_ballot_node": self._accepted_ballot.node_id if self._accepted_ballot else None,
                "accepted_value": self._accepted_value,
            },
            daemon=True,
        )
        return [promise]

    def _handle_promise(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot_number = metadata["ballot_number"]

        if ballot_number not in self._phase1_responses:
            return []

        accepted_ballot = None
        if metadata.get("accepted_ballot_number") is not None:
            accepted_ballot = (metadata["accepted_ballot_number"], metadata["accepted_ballot_node"])

        response = {
            "from": metadata.get("from"),
            "accepted_ballot": accepted_ballot,
            "accepted_value": metadata.get("accepted_value"),
        }

        self._phase1_responses[ballot_number].append(response)
        self._promises_received += 1

        # Check if we have a quorum
        if len(self._phase1_responses[ballot_number]) >= self.quorum_size:
            return self._start_phase2(ballot_number)

        return []

    def _handle_nack(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot_number = metadata.get("ballot_number")
        highest_number = metadata.get("highest_ballot_number", 0)
        self._nacks_received += 1

        # Update our knowledge of highest ballot
        if highest_number > self._current_ballot.number:
            self._current_ballot = Ballot(highest_number, self.name)

        # Schedule retry with higher ballot
        if ballot_number in self._proposed_values:
            retry = Event(
                time=self.now + self._retry_delay * (1 + random.random()),
                event_type="PaxosRetry",
                target=self,
                daemon=True,
                context={"metadata": {"original_ballot": ballot_number}},
            )
            return [retry]
        return []

    def _handle_retry(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        original_ballot = metadata.get("original_ballot")

        if self._decided:
            return []

        if original_ballot in self._proposed_values:
            value = self._proposed_values[original_ballot]
            future = self._proposal_futures.get(original_ballot)

            # Create new ballot
            new_number = self._current_ballot.number + 1
            self._current_ballot = Ballot(new_number, self.name)

            # Transfer future to new ballot
            if future:
                self._proposal_futures[new_number] = future
                del self._proposal_futures[original_ballot]
            self._proposed_values[new_number] = value
            del self._proposed_values[original_ballot]
            self._phase1_responses[new_number] = []
            self._phase2_responses[new_number] = 0

            return self.start_phase1()
        return []

    def _start_phase2(self, ballot_number: int) -> list[Event]:
        """Begin Phase 2 with the value determined from Phase 1 responses."""
        responses = self._phase1_responses[ballot_number]

        # Find the highest-ballot accepted value among promises
        highest_accepted_ballot = None
        chosen_value = self._proposed_values.get(ballot_number)

        for resp in responses:
            ab = resp.get("accepted_ballot")
            if ab is not None:
                if highest_accepted_ballot is None or ab > highest_accepted_ballot:
                    highest_accepted_ballot = ab
                    chosen_value = resp["accepted_value"]

        ballot = Ballot(ballot_number, self.name)
        # Update proposed value with consensus value
        self._proposed_values[ballot_number] = chosen_value

        # Self-accept
        if self._promised_ballot is None or ballot >= self._promised_ballot:
            self._accepted_ballot = ballot
            self._accepted_value = chosen_value
            self._phase2_responses[ballot_number] = 1  # count self

        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="PaxosAccept",
                payload={
                    "ballot_number": ballot_number,
                    "ballot_node": self.name,
                    "value": chosen_value,
                },
                daemon=True,
            )
            events.append(msg)

        # Check if self-accept alone is enough (e.g., single node)
        if self._phase2_responses.get(ballot_number, 0) >= self.quorum_size:
            events.extend(self._decide(ballot_number, chosen_value))

        return events

    def _handle_accept(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot = Ballot(metadata["ballot_number"], metadata["ballot_node"])
        value = metadata["value"]

        sender = self._find_peer(metadata.get("source"))
        if sender is None:
            return []

        if self._promised_ballot is not None and ballot < self._promised_ballot:
            nack = self._network.send(
                source=self,
                destination=sender,
                event_type="PaxosNack",
                payload={
                    "ballot_number": ballot.number,
                    "ballot_node": ballot.node_id,
                    "highest_ballot_number": self._promised_ballot.number,
                    "highest_ballot_node": self._promised_ballot.node_id,
                },
                daemon=True,
            )
            return [nack]

        # Accept
        self._promised_ballot = ballot
        self._accepted_ballot = ballot
        self._accepted_value = value

        accepted = self._network.send(
            source=self,
            destination=sender,
            event_type="PaxosAccepted",
            payload={
                "ballot_number": ballot.number,
                "ballot_node": ballot.node_id,
                "from": self.name,
            },
            daemon=True,
        )
        return [accepted]

    def _handle_accepted(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot_number = metadata["ballot_number"]
        self._accepts_received += 1

        if ballot_number not in self._phase2_responses:
            self._phase2_responses[ballot_number] = 0
        self._phase2_responses[ballot_number] += 1

        if self._phase2_responses[ballot_number] >= self.quorum_size and not self._decided:
            value = self._proposed_values.get(ballot_number)
            return self._decide(ballot_number, value)
        return []

    def _handle_decided(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        value = metadata.get("value")
        if not self._decided:
            self._decided = True
            self._decided_value = value
            logger.debug("[%s] Learned decided value: %r", self.name, value)
        return None

    def _decide(self, ballot_number: int, value: Any) -> list[Event]:
        if self._decided:
            return []

        self._decided = True
        self._decided_value = value
        self._proposals_succeeded += 1
        logger.debug("[%s] Decided value: %r (ballot=%d)", self.name, value, ballot_number)

        # Resolve the proposal future
        future = self._proposal_futures.get(ballot_number)
        if future:
            future.resolve(value)

        # Broadcast decision to all peers
        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="PaxosDecided",
                payload={"value": value},
                daemon=True,
            )
            events.append(msg)

        return events

    def _find_peer(self, source_name: str | None) -> Entity | None:
        """Find a peer entity by name from network routing metadata."""
        if source_name is None:
            return None
        for peer in self._peers:
            if peer.name == source_name:
                return peer
        return None

    @property
    def stats(self) -> PaxosStats:
        return PaxosStats(
            proposals_started=self._proposals_started,
            proposals_succeeded=self._proposals_succeeded,
            proposals_failed=self._proposals_failed,
            promises_received=self._promises_received,
            nacks_received=self._nacks_received,
            accepts_received=self._accepts_received,
            decided_value=self._decided_value,
        )

    def __repr__(self) -> str:
        return (
            f"PaxosNode({self.name}, decided={self._decided}, "
            f"ballot={self._current_ballot})"
        )
