"""Multi-decree Paxos (Multi-Paxos) consensus protocol.

Extends single-decree Paxos to a sequence of slots (log), enabling
continuous consensus on a stream of commands. Includes stable leader
optimization to skip Phase 1 when a leader is established.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.components.consensus.log import Log, LogEntry
from happysimulator.components.consensus.paxos import Ballot
from happysimulator.components.consensus.raft_state_machine import StateMachine, KVStateMachine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiPaxosStats:
    """Statistics snapshot from a MultiPaxosNode.

    Attributes:
        is_leader: Whether this node believes it is the leader.
        current_ballot: Current ballot number.
        log_length: Number of entries in the log.
        commit_index: Highest committed log index.
        commands_committed: Total commands committed.
        leader_changes: Number of leader changes observed.
    """
    is_leader: bool
    current_ballot: int
    log_length: int
    commit_index: int
    commands_committed: int
    leader_changes: int


class MultiPaxosNode(Entity):
    """Multi-decree Paxos participant with stable leader optimization.

    Args:
        name: Node identifier.
        network: Network for communication.
        peers: List of peer nodes.
        state_machine: State machine to apply committed commands.
        leader_lease_timeout: Seconds before leader lease expires.
        heartbeat_interval: Seconds between leader heartbeats.
    """

    def __init__(
        self,
        name: str,
        network: Any,
        peers: list[MultiPaxosNode] | None = None,
        state_machine: StateMachine | None = None,
        leader_lease_timeout: float = 5.0,
        heartbeat_interval: float = 1.0,
    ) -> None:
        super().__init__(name)
        self._network = network
        self._peers: list[MultiPaxosNode] = list(peers) if peers else []
        self._state_machine = state_machine or KVStateMachine()
        self._leader_lease_timeout = leader_lease_timeout
        self._heartbeat_interval = heartbeat_interval

        # Log
        self._log = Log()
        self._last_applied: int = 0

        # Ballot/leader
        self._current_ballot = Ballot(0, self.name)
        self._leader: str | None = None
        self._is_leader: bool = False
        self._leader_established: bool = False  # stable leader opt
        self._last_leader_heartbeat: float = 0.0

        # Per-slot tracking
        self._slot_futures: dict[int, SimFuture] = {}  # slot -> future
        self._slot_commands: dict[int, Any] = {}  # slot -> command
        self._slot_acks: dict[int, int] = {}  # slot -> ack count
        self._pending_commands: list[tuple[Any, SimFuture]] = []

        # Phase 1 state
        self._phase1_responses: dict[int, list[dict]] = {}

        # Heartbeat event
        self._heartbeat_event: Event | None = None

        # Stats
        self._commands_committed: int = 0
        self._leader_changes: int = 0

    def set_peers(self, peers: list[MultiPaxosNode]) -> None:
        self._peers = [p for p in peers if p.name != self.name]

    @property
    def quorum_size(self) -> int:
        total = len(self._peers) + 1
        return (total // 2) + 1

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
        """Submit a command for consensus.

        Returns a SimFuture resolving with (index, result) on commit.
        """
        future = SimFuture()

        if not self._is_leader:
            # Forward to leader if known
            if self._leader and self._leader != self.name:
                self._pending_commands.append((command, future))
                # Will be forwarded on next event processing
            else:
                self._pending_commands.append((command, future))
            return future

        self._assign_slot(command, future)
        return future

    def _assign_slot(self, command: Any, future: SimFuture) -> None:
        slot = self._log.last_index + 1
        self._log.append(self._current_ballot.number, command)
        self._slot_futures[slot] = future
        self._slot_commands[slot] = command
        self._slot_acks[slot] = 1  # self

    def start(self) -> list[Event]:
        """Start by attempting to become leader."""
        return self._begin_phase1()

    def _begin_phase1(self) -> list[Event]:
        """Send Prepare to establish leadership."""
        max_seen = self._current_ballot.number
        self._current_ballot = Ballot(max_seen + 1, self.name)
        ballot = self._current_ballot
        self._phase1_responses[ballot.number] = []

        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="MultiPaxosPrepare",
                payload={
                    "ballot_number": ballot.number,
                    "ballot_node": ballot.node_id,
                    "log_length": self._log.last_index,
                },
                daemon=True,
            )
            events.append(msg)

        # Self-promise
        self._phase1_responses[ballot.number].append({
            "from": self.name,
            "log_entries": [],
        })

        if len(self._phase1_responses[ballot.number]) >= self.quorum_size:
            events.extend(self._become_leader())

        return events

    def handle_event(self, event: Event):
        handlers = {
            "MultiPaxosPrepare": self._handle_prepare,
            "MultiPaxosPromise": self._handle_promise,
            "MultiPaxosAccept": self._handle_accept,
            "MultiPaxosAccepted": self._handle_accepted,
            "MultiPaxosHeartbeat": self._handle_heartbeat,
            "MultiPaxosForward": self._handle_forward,
            "MultiPaxosNack": self._handle_nack,
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
            nack = self._network.send(
                source=self,
                destination=sender,
                event_type="MultiPaxosNack",
                payload={
                    "ballot_number": self._current_ballot.number,
                    "ballot_node": self._current_ballot.node_id,
                },
                daemon=True,
            )
            return [nack]

        self._current_ballot = ballot
        self._is_leader = False

        # Send our log entries
        entries = [
            {"index": e.index, "term": e.term, "command": e.command}
            for e in self._log.entries_after(0)
        ]
        promise = self._network.send(
            source=self,
            destination=sender,
            event_type="MultiPaxosPromise",
            payload={
                "ballot_number": ballot.number,
                "ballot_node": ballot.node_id,
                "from": self.name,
                "log_entries": entries,
                "commit_index": self._log.commit_index,
            },
            daemon=True,
        )
        return [promise]

    def _handle_promise(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        ballot_number = metadata["ballot_number"]

        if ballot_number not in self._phase1_responses:
            return []

        self._phase1_responses[ballot_number].append({
            "from": metadata.get("from"),
            "log_entries": metadata.get("log_entries", []),
            "commit_index": metadata.get("commit_index", 0),
        })

        if len(self._phase1_responses[ballot_number]) >= self.quorum_size:
            return self._become_leader()
        return []

    def _become_leader(self) -> list[Event]:
        self._is_leader = True
        self._leader = self.name
        self._leader_established = True
        self._leader_changes += 1
        self._last_leader_heartbeat = self.now.to_seconds()
        logger.debug("[%s] Became leader (ballot=%s)", self.name, self._current_ballot)

        events: list[Event] = []

        # Process pending commands
        for command, future in self._pending_commands:
            self._assign_slot(command, future)
        self._pending_commands.clear()

        # Send initial heartbeat
        events.extend(self._send_heartbeat())

        # Replicate uncommitted entries
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
            nack = self._network.send(
                source=self,
                destination=sender,
                event_type="MultiPaxosNack",
                payload={
                    "ballot_number": self._current_ballot.number,
                    "ballot_node": self._current_ballot.node_id,
                },
                daemon=True,
            )
            return [nack]

        self._current_ballot = ballot
        self._leader = ballot.node_id

        # Append to log (truncate conflicting entries)
        if slot > self._log.last_index:
            self._log.append(ballot.number, command)
        elif self._log.get(slot) and self._log.get(slot).term != ballot.number:
            self._log.truncate_from(slot)
            self._log.append(ballot.number, command)

        # Advance commit index
        leader_commit = metadata.get("commit_index", 0)
        if leader_commit > self._log.commit_index:
            newly_committed = self._log.advance_commit(leader_commit)
            self._apply_committed(newly_committed)

        accepted = self._network.send(
            source=self,
            destination=sender,
            event_type="MultiPaxosAccepted",
            payload={
                "ballot_number": ballot.number,
                "slot": slot,
                "from": self.name,
            },
            daemon=True,
        )
        return [accepted]

    def _handle_accepted(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        slot = metadata["slot"]

        if slot not in self._slot_acks:
            self._slot_acks[slot] = 0
        self._slot_acks[slot] += 1

        if self._slot_acks[slot] >= self.quorum_size:
            # Commit this slot
            if slot > self._log.commit_index:
                newly_committed = self._log.advance_commit(slot)
                self._apply_committed(newly_committed)
        return []

    def _handle_heartbeat(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        ballot = Ballot(metadata.get("ballot_number", 0), metadata.get("ballot_node", ""))
        leader_commit = metadata.get("commit_index", 0)

        if ballot >= self._current_ballot:
            self._current_ballot = ballot
            self._leader = ballot.node_id
            self._is_leader = False
            self._last_leader_heartbeat = self.now.to_seconds()

            if leader_commit > self._log.commit_index:
                newly_committed = self._log.advance_commit(leader_commit)
                self._apply_committed(newly_committed)
        return None

    def _handle_forward(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        command = metadata.get("command")
        # If we're the leader, process it
        if self._is_leader and command is not None:
            future = SimFuture()
            self._assign_slot(command, future)
            slot = self._log.last_index
            return self._replicate_slot(slot)
        return []

    def _handle_nack(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        higher_ballot = Ballot(
            metadata.get("ballot_number", 0),
            metadata.get("ballot_node", ""),
        )
        if higher_ballot > self._current_ballot:
            self._current_ballot = higher_ballot
            self._is_leader = False
            self._leader_established = False
        return []

    def _replicate_slot(self, slot: int) -> list[Event]:
        entry = self._log.get(slot)
        if entry is None:
            return []

        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="MultiPaxosAccept",
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
                source=self,
                destination=peer,
                event_type="MultiPaxosHeartbeat",
                payload={
                    "ballot_number": self._current_ballot.number,
                    "ballot_node": self._current_ballot.node_id,
                    "commit_index": self._log.commit_index,
                },
                daemon=True,
            )
            events.append(msg)

        # Schedule next heartbeat
        hb = Event(
            time=self.now + self._heartbeat_interval,
            event_type="MultiPaxosHeartbeat",
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

                # Resolve future if we have one
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
    def stats(self) -> MultiPaxosStats:
        return MultiPaxosStats(
            is_leader=self._is_leader,
            current_ballot=self._current_ballot.number,
            log_length=self._log.last_index,
            commit_index=self._log.commit_index,
            commands_committed=self._commands_committed,
            leader_changes=self._leader_changes,
        )

    def __repr__(self) -> str:
        return (
            f"MultiPaxosNode({self.name}, "
            f"leader={self._is_leader}, "
            f"ballot={self._current_ballot})"
        )
