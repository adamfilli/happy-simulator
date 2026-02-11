"""Flexible Paxos — asymmetric quorum variant of Multi-Paxos.

Flexible Paxos relaxes the classic Paxos quorum requirement by allowing
different quorum sizes for Phase 1 (prepare) and Phase 2 (accept),
subject to the constraint: Q1 + Q2 > N.

This enables tuning for different workloads:
- Small Q2 = fast writes (majority of operations)
- Small Q1 = fast leader recovery
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.components.consensus.log import Log, LogEntry
from happysimulator.components.consensus.paxos import Ballot
from happysimulator.components.consensus.raft_state_machine import StateMachine, KVStateMachine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlexiblePaxosStats:
    """Statistics snapshot from a FlexiblePaxosNode.

    Attributes:
        is_leader: Whether this node is the leader.
        current_ballot: Current ballot number.
        log_length: Number of entries in the log.
        commit_index: Highest committed log index.
        commands_committed: Total commands committed.
        phase1_quorum: Phase 1 quorum size.
        phase2_quorum: Phase 2 quorum size.
    """
    is_leader: bool
    current_ballot: int
    log_length: int
    commit_index: int
    commands_committed: int
    phase1_quorum: int
    phase2_quorum: int


class FlexiblePaxosNode(Entity):
    """Flexible Paxos participant with asymmetric quorums.

    Args:
        name: Node identifier.
        network: Network for communication.
        peers: List of peer nodes.
        state_machine: State machine for committed commands.
        phase1_quorum: Quorum size for Phase 1 (prepare).
        phase2_quorum: Quorum size for Phase 2 (accept).
        heartbeat_interval: Seconds between leader heartbeats.

    Raises:
        ValueError: If quorum sizes violate Q1 + Q2 > N.
    """

    def __init__(
        self,
        name: str,
        network: Any,
        peers: list[FlexiblePaxosNode] | None = None,
        state_machine: StateMachine | None = None,
        phase1_quorum: int | None = None,
        phase2_quorum: int | None = None,
        heartbeat_interval: float = 1.0,
    ) -> None:
        super().__init__(name)
        self._network = network
        self._peers: list[FlexiblePaxosNode] = list(peers) if peers else []
        self._state_machine = state_machine or KVStateMachine()
        self._heartbeat_interval = heartbeat_interval

        total = len(self._peers) + 1
        self._phase1_quorum = phase1_quorum if phase1_quorum is not None else (total // 2) + 1
        self._phase2_quorum = phase2_quorum if phase2_quorum is not None else (total // 2) + 1

        if self._phase1_quorum + self._phase2_quorum <= total:
            raise ValueError(
                f"Quorum sizes must satisfy Q1 + Q2 > N: "
                f"{self._phase1_quorum} + {self._phase2_quorum} <= {total}"
            )

        # Log
        self._log = Log()
        self._last_applied: int = 0

        # Ballot/leader
        self._current_ballot = Ballot(0, self.name)
        self._leader: str | None = None
        self._is_leader: bool = False

        # Per-slot tracking
        self._slot_futures: dict[int, SimFuture] = {}
        self._slot_acks: dict[int, int] = {}
        self._pending_commands: list[tuple[Any, SimFuture]] = []

        # Phase 1 state
        self._phase1_responses: dict[int, list[dict]] = {}

        # Heartbeat
        self._heartbeat_event: Event | None = None

        # Stats
        self._commands_committed: int = 0

    def set_peers(self, peers: list[FlexiblePaxosNode]) -> None:
        self._peers = [p for p in peers if p.name != self.name]
        total = len(self._peers) + 1
        if self._phase1_quorum + self._phase2_quorum <= total:
            raise ValueError(
                f"Quorum sizes must satisfy Q1 + Q2 > N: "
                f"{self._phase1_quorum} + {self._phase2_quorum} <= {total}"
            )

    @property
    def phase1_quorum(self) -> int:
        return self._phase1_quorum

    @property
    def phase2_quorum(self) -> int:
        return self._phase2_quorum

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    @property
    def leader(self) -> str | None:
        return self._leader

    @property
    def log(self) -> Log:
        return self._log

    def submit(self, command: Any) -> SimFuture:
        """Submit a command for consensus."""
        future = SimFuture()
        if not self._is_leader:
            self._pending_commands.append((command, future))
            return future
        self._assign_slot(command, future)
        return future

    def _assign_slot(self, command: Any, future: SimFuture) -> None:
        slot = self._log.last_index + 1
        self._log.append(self._current_ballot.number, command)
        self._slot_futures[slot] = future
        self._slot_acks[slot] = 1  # self

    def start(self) -> list[Event]:
        return self._begin_phase1()

    def _begin_phase1(self) -> list[Event]:
        self._current_ballot = Ballot(self._current_ballot.number + 1, self.name)
        ballot = self._current_ballot
        self._phase1_responses[ballot.number] = [{"from": self.name}]

        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="FlexPaxosPrepare",
                payload={
                    "ballot_number": ballot.number,
                    "ballot_node": ballot.node_id,
                },
                daemon=True,
            )
            events.append(msg)

        if len(self._phase1_responses[ballot.number]) >= self._phase1_quorum:
            events.extend(self._become_leader())

        return events

    def handle_event(self, event: Event):
        handlers = {
            "FlexPaxosPrepare": self._handle_prepare,
            "FlexPaxosPromise": self._handle_promise,
            "FlexPaxosAccept": self._handle_accept,
            "FlexPaxosAccepted": self._handle_accepted,
            "FlexPaxosHeartbeat": self._handle_heartbeat,
            "FlexPaxosNack": self._handle_nack,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        return None

    def _handle_prepare(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot = Ballot(metadata["ballot_number"], metadata["ballot_node"])
        sender = self._find_peer(metadata.get("source"))

        if sender is None:
            return []

        if self._current_ballot > ballot:
            return [self._network.send(
                source=self, destination=sender,
                event_type="FlexPaxosNack",
                payload={
                    "ballot_number": self._current_ballot.number,
                    "ballot_node": self._current_ballot.node_id,
                },
                daemon=True,
            )]

        self._current_ballot = ballot
        self._is_leader = False

        entries = [
            {"index": e.index, "term": e.term, "command": e.command}
            for e in self._log.entries_after(0)
        ]
        return [self._network.send(
            source=self, destination=sender,
            event_type="FlexPaxosPromise",
            payload={
                "ballot_number": ballot.number,
                "ballot_node": ballot.node_id,
                "from": self.name,
                "log_entries": entries,
                "commit_index": self._log.commit_index,
            },
            daemon=True,
        )]

    def _handle_promise(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot_number = metadata["ballot_number"]

        if ballot_number not in self._phase1_responses:
            return []

        self._phase1_responses[ballot_number].append({
            "from": metadata.get("from"),
            "log_entries": metadata.get("log_entries", []),
        })

        if len(self._phase1_responses[ballot_number]) >= self._phase1_quorum:
            return self._become_leader()
        return []

    def _become_leader(self) -> list[Event]:
        self._is_leader = True
        self._leader = self.name
        logger.debug("[%s] Became FlexPaxos leader (ballot=%s, Q1=%d, Q2=%d)",
                     self.name, self._current_ballot, self._phase1_quorum, self._phase2_quorum)

        events: list[Event] = []
        for command, future in self._pending_commands:
            self._assign_slot(command, future)
        self._pending_commands.clear()

        events.extend(self._send_heartbeat())
        for slot_idx in range(self._log.commit_index + 1, self._log.last_index + 1):
            events.extend(self._replicate_slot(slot_idx))
        return events

    def _handle_accept(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot = Ballot(metadata["ballot_number"], metadata["ballot_node"])
        slot = metadata["slot"]
        command = metadata["command"]
        sender = self._find_peer(metadata.get("source"))

        if sender is None:
            return []

        if ballot < self._current_ballot:
            return [self._network.send(
                source=self, destination=sender,
                event_type="FlexPaxosNack",
                payload={
                    "ballot_number": self._current_ballot.number,
                    "ballot_node": self._current_ballot.node_id,
                },
                daemon=True,
            )]

        self._current_ballot = ballot
        self._leader = ballot.node_id

        if slot > self._log.last_index:
            self._log.append(ballot.number, command)
        elif self._log.get(slot) and self._log.get(slot).term != ballot.number:
            self._log.truncate_from(slot)
            self._log.append(ballot.number, command)

        leader_commit = metadata.get("commit_index", 0)
        if leader_commit > self._log.commit_index:
            newly_committed = self._log.advance_commit(leader_commit)
            self._apply_committed(newly_committed)

        return [self._network.send(
            source=self, destination=sender,
            event_type="FlexPaxosAccepted",
            payload={
                "ballot_number": ballot.number,
                "slot": slot,
                "from": self.name,
            },
            daemon=True,
        )]

    def _handle_accepted(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        slot = metadata["slot"]

        if slot not in self._slot_acks:
            self._slot_acks[slot] = 0
        self._slot_acks[slot] += 1

        if self._slot_acks[slot] >= self._phase2_quorum:
            if slot > self._log.commit_index:
                newly_committed = self._log.advance_commit(slot)
                self._apply_committed(newly_committed)
        return []

    def _handle_heartbeat(self, event: Event) -> list[Event] | None:
        metadata = event.context.get("metadata", {})

        # Self-heartbeat tick: send heartbeats to peers
        if metadata.get("self_heartbeat"):
            if not self._is_leader:
                return None
            return self._send_heartbeat()

        ballot = Ballot(metadata.get("ballot_number", 0), metadata.get("ballot_node", ""))
        leader_commit = metadata.get("commit_index", 0)

        if ballot >= self._current_ballot:
            self._current_ballot = ballot
            self._leader = ballot.node_id
            self._is_leader = False

            if leader_commit > self._log.commit_index:
                newly_committed = self._log.advance_commit(leader_commit)
                self._apply_committed(newly_committed)
        return None

    def _handle_nack(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        higher = Ballot(metadata.get("ballot_number", 0), metadata.get("ballot_node", ""))
        if higher > self._current_ballot:
            self._current_ballot = higher
            self._is_leader = False
        return []

    def _replicate_slot(self, slot: int) -> list[Event]:
        entry = self._log.get(slot)
        if entry is None:
            return []

        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self, destination=peer,
                event_type="FlexPaxosAccept",
                payload={
                    "ballot_number": self._current_ballot.number,
                    "ballot_node": self._current_ballot.node_id,
                    "slot": slot,
                    "command": entry.command,
                    "commit_index": self._log.commit_index,
                },
                daemon=True,
            )
            events.append(msg)
        return events

    def _send_heartbeat(self) -> list[Event]:
        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self, destination=peer,
                event_type="FlexPaxosHeartbeat",
                payload={
                    "ballot_number": self._current_ballot.number,
                    "ballot_node": self._current_ballot.node_id,
                    "commit_index": self._log.commit_index,
                },
                daemon=True,
            )
            events.append(msg)

        hb = Event(
            time=self.now + self._heartbeat_interval,
            event_type="FlexPaxosHeartbeat",
            target=self,
            daemon=True,
            context={"metadata": {
                "ballot_number": self._current_ballot.number,
                "ballot_node": self._current_ballot.node_id,
                "commit_index": self._log.commit_index,
                "self_heartbeat": True,
            }},
        )
        if self._heartbeat_event:
            self._heartbeat_event.cancel()
        self._heartbeat_event = hb
        events.append(hb)
        return events

    def _apply_committed(self, entries: list[LogEntry]) -> None:
        for entry in entries:
            if entry.index > self._last_applied:
                result = self._state_machine.apply(entry.command)
                self._last_applied = entry.index
                self._commands_committed += 1

                future = self._slot_futures.pop(entry.index, None)
                if future:
                    future.resolve((entry.index, result))

    def _find_peer(self, source_name: str | None) -> Entity | None:
        if source_name is None:
            return None
        for peer in self._peers:
            if peer.name == source_name:
                return peer
        return None

    @property
    def stats(self) -> FlexiblePaxosStats:
        return FlexiblePaxosStats(
            is_leader=self._is_leader,
            current_ballot=self._current_ballot.number,
            log_length=self._log.last_index,
            commit_index=self._log.commit_index,
            commands_committed=self._commands_committed,
            phase1_quorum=self._phase1_quorum,
            phase2_quorum=self._phase2_quorum,
        )

    def __repr__(self) -> str:
        return (
            f"FlexiblePaxosNode({self.name}, leader={self._is_leader}, "
            f"Q1={self._phase1_quorum}, Q2={self._phase2_quorum})"
        )
