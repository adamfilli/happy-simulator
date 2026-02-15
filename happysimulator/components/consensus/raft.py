"""Raft consensus protocol implementation.

Full Raft with leader election, log replication, and commitment.
Each RaftNode transitions between Follower, Candidate, and Leader states
using randomized election timeouts and AppendEntries heartbeats.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from happysimulator.components.consensus.log import Log, LogEntry
from happysimulator.components.consensus.raft_state_machine import KVStateMachine, StateMachine
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture

logger = logging.getLogger(__name__)


class RaftState(Enum):
    """Raft node states."""

    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass(frozen=True)
class RaftStats:
    """Statistics snapshot from a RaftNode.

    Attributes:
        state: Current node state.
        current_term: Current term number.
        current_leader: Name of the known leader, or None.
        log_length: Number of entries in the log.
        commit_index: Highest committed index.
        commands_committed: Total commands committed via this node.
        elections_started: Total elections initiated.
        votes_received: Total votes received in current/past elections.
    """

    state: RaftState = RaftState.FOLLOWER
    current_term: int = 0
    current_leader: str | None = None
    log_length: int = 0
    commit_index: int = 0
    commands_committed: int = 0
    elections_started: int = 0
    votes_received: int = 0


class RaftNode(Entity):
    """Raft consensus participant.

    Args:
        name: Node identifier.
        network: Network for communication.
        peers: List of peer RaftNode entities.
        state_machine: State machine to apply committed commands.
        election_timeout_min: Minimum election timeout in seconds.
        election_timeout_max: Maximum election timeout in seconds.
        heartbeat_interval: Seconds between leader heartbeats.
    """

    def __init__(
        self,
        name: str,
        network: Any,
        peers: list[RaftNode] | None = None,
        state_machine: StateMachine | None = None,
        election_timeout_min: float = 1.5,
        election_timeout_max: float = 3.0,
        heartbeat_interval: float = 0.5,
    ) -> None:
        super().__init__(name)
        self._network = network
        self._peers: list[RaftNode] = list(peers) if peers else []
        self._state_machine = state_machine or KVStateMachine()
        self._election_timeout_min = election_timeout_min
        self._election_timeout_max = election_timeout_max
        self._heartbeat_interval = heartbeat_interval

        # Persistent state
        self._current_term: int = 0
        self._voted_for: str | None = None
        self._log = Log()

        # Volatile state
        self._state = RaftState.FOLLOWER
        self._leader: str | None = None
        self._last_applied: int = 0

        # Leader state
        self._next_index: dict[str, int] = {}
        self._match_index: dict[str, int] = {}

        # Election state
        self._votes_received_set: set[str] = set()
        self._election_timeout_event: Event | None = None
        self._heartbeat_event: Event | None = None

        # Pending client requests
        self._pending_futures: dict[int, SimFuture] = {}  # log_index -> future

        # Stats
        self._commands_committed: int = 0
        self._elections_started: int = 0
        self._total_votes_received: int = 0

    def set_peers(self, peers: list[RaftNode]) -> None:
        self._peers = [p for p in peers if p.name != self.name]

    @property
    def quorum_size(self) -> int:
        total = len(self._peers) + 1
        return (total // 2) + 1

    @property
    def state(self) -> RaftState:
        return self._state

    @property
    def current_term(self) -> int:
        return self._current_term

    @property
    def current_leader(self) -> str | None:
        return self._leader

    @property
    def is_leader(self) -> bool:
        return self._state == RaftState.LEADER

    @property
    def log(self) -> Log:
        return self._log

    def submit(self, command: Any) -> SimFuture:
        """Submit a command for consensus.

        Returns a SimFuture resolving with (index, result) on commit.
        """
        future = SimFuture()

        if self._state != RaftState.LEADER:
            # Not leader — queue for forwarding
            self._pending_futures[-(len(self._pending_futures) + 1)] = future
            return future

        entry = self._log.append(self._current_term, command)
        self._pending_futures[entry.index] = future
        return future

    def start(self) -> list[Event]:
        """Start the Raft node by scheduling initial election timeout."""
        return [self._schedule_election_timeout()]

    def handle_event(self, event: Event):
        handlers = {
            "RaftElectionTimeout": self._handle_election_timeout,
            "RaftRequestVote": self._handle_request_vote,
            "RaftVoteResponse": self._handle_vote_response,
            "RaftAppendEntries": self._handle_append_entries,
            "RaftAppendEntriesResponse": self._handle_append_entries_response,
            "RaftHeartbeat": self._handle_heartbeat_tick,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        return None

    def _schedule_election_timeout(self) -> Event:
        if self._election_timeout_event:
            self._election_timeout_event.cancel()
        timeout = random.uniform(self._election_timeout_min, self._election_timeout_max)
        evt = Event(
            time=self.now + timeout,
            event_type="RaftElectionTimeout",
            target=self,
            daemon=True,
        )
        self._election_timeout_event = evt
        return evt

    def _schedule_heartbeat(self) -> Event:
        if self._heartbeat_event:
            self._heartbeat_event.cancel()
        evt = Event(
            time=self.now + self._heartbeat_interval,
            event_type="RaftHeartbeat",
            target=self,
            daemon=True,
        )
        self._heartbeat_event = evt
        return evt

    # ── Election ──

    def _handle_election_timeout(self, event: Event) -> list[Event]:
        if event.cancelled:
            return []

        # Don't start election if we're leader
        if self._state == RaftState.LEADER:
            return [self._schedule_election_timeout()]

        return self._start_election()

    def _start_election(self) -> list[Event]:
        self._state = RaftState.CANDIDATE
        self._current_term += 1
        self._voted_for = self.name
        self._votes_received_set = {self.name}
        self._leader = None
        self._elections_started += 1
        self._total_votes_received += 1

        logger.debug(
            "[%s] Starting election for term %d",
            self.name,
            self._current_term,
        )

        events: list[Event] = []
        for peer in self._peers:
            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="RaftRequestVote",
                payload={
                    "term": self._current_term,
                    "candidate_id": self.name,
                    "last_log_index": self._log.last_index,
                    "last_log_term": self._log.last_term,
                },
                daemon=True,
            )
            events.append(msg)

        # Check if we already have quorum (single node)
        if len(self._votes_received_set) >= self.quorum_size:
            events.extend(self._become_leader())
        else:
            events.append(self._schedule_election_timeout())

        return events

    def _handle_request_vote(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        term = metadata["term"]
        candidate = metadata["candidate_id"]
        last_log_index = metadata.get("last_log_index", 0)
        last_log_term = metadata.get("last_log_term", 0)

        sender = self._find_peer(metadata.get("source"))
        if sender is None:
            return []

        if term > self._current_term:
            self._step_down(term)

        vote_granted = False
        if (
            term >= self._current_term
            and (self._voted_for is None or self._voted_for == candidate)
            and (
                last_log_term > self._log.last_term
                or (last_log_term == self._log.last_term and last_log_index >= self._log.last_index)
            )
        ):
            vote_granted = True
            self._voted_for = candidate
            self._current_term = term

        resp = self._network.send(
            source=self,
            destination=sender,
            event_type="RaftVoteResponse",
            payload={
                "term": self._current_term,
                "vote_granted": vote_granted,
                "from": self.name,
            },
            daemon=True,
        )

        events = [resp]
        if vote_granted:
            # Reset election timeout since we granted a vote
            events.append(self._schedule_election_timeout())
        return events

    def _handle_vote_response(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        term = metadata["term"]
        granted = metadata["vote_granted"]
        voter = metadata.get("from")

        if term > self._current_term:
            self._step_down(term)
            return [self._schedule_election_timeout()]

        if self._state != RaftState.CANDIDATE or term != self._current_term:
            return []

        if granted and voter:
            self._votes_received_set.add(voter)
            self._total_votes_received += 1

        if len(self._votes_received_set) >= self.quorum_size:
            return self._become_leader()
        return []

    def _become_leader(self) -> list[Event]:
        self._state = RaftState.LEADER
        self._leader = self.name
        logger.debug("[%s] Became leader for term %d", self.name, self._current_term)

        # Initialize leader state
        for peer in self._peers:
            self._next_index[peer.name] = self._log.last_index + 1
            self._match_index[peer.name] = 0

        # Cancel election timeout
        if self._election_timeout_event:
            self._election_timeout_event.cancel()

        # Send initial heartbeats
        events = self._send_append_entries()
        events.append(self._schedule_heartbeat())
        return events

    def _step_down(self, new_term: int) -> None:
        self._current_term = new_term
        self._state = RaftState.FOLLOWER
        self._voted_for = None
        if self._heartbeat_event:
            self._heartbeat_event.cancel()
            self._heartbeat_event = None

    # ── Replication ──

    def _handle_heartbeat_tick(self, event: Event) -> list[Event]:
        if event.cancelled:
            return []

        if self._state != RaftState.LEADER:
            return [self._schedule_election_timeout()]

        events = self._send_append_entries()
        events.append(self._schedule_heartbeat())
        return events

    def _send_append_entries(self) -> list[Event]:
        events: list[Event] = []
        for peer in self._peers:
            prev_log_index = self._next_index.get(peer.name, 1) - 1
            prev_log_term = 0
            if prev_log_index > 0:
                prev_entry = self._log.get(prev_log_index)
                if prev_entry:
                    prev_log_term = prev_entry.term

            entries = self._log.entries_after(prev_log_index)
            entry_dicts = [
                {"index": e.index, "term": e.term, "command": e.command} for e in entries
            ]

            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="RaftAppendEntries",
                payload={
                    "term": self._current_term,
                    "leader_id": self.name,
                    "prev_log_index": prev_log_index,
                    "prev_log_term": prev_log_term,
                    "entries": entry_dicts,
                    "leader_commit": self._log.commit_index,
                },
                daemon=True,
            )
            events.append(msg)
        return events

    def _handle_append_entries(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        term = metadata["term"]
        leader_id = metadata["leader_id"]
        prev_log_index = metadata.get("prev_log_index", 0)
        prev_log_term = metadata.get("prev_log_term", 0)
        entries = metadata.get("entries", [])
        leader_commit = metadata.get("leader_commit", 0)

        sender = self._find_peer(metadata.get("source"))
        if sender is None:
            return []

        if term < self._current_term:
            resp = self._network.send(
                source=self,
                destination=sender,
                event_type="RaftAppendEntriesResponse",
                payload={
                    "term": self._current_term,
                    "success": False,
                    "from": self.name,
                    "match_index": 0,
                },
                daemon=True,
            )
            return [resp]

        if term >= self._current_term:
            self._step_down(term)
        self._leader = leader_id
        self._current_term = term

        # Reset election timeout
        result_events: list[Event] = [self._schedule_election_timeout()]

        # Log consistency check
        if prev_log_index > 0:
            prev_entry = self._log.get(prev_log_index)
            if prev_entry is None or prev_entry.term != prev_log_term:
                resp = self._network.send(
                    source=self,
                    destination=sender,
                    event_type="RaftAppendEntriesResponse",
                    payload={
                        "term": self._current_term,
                        "success": False,
                        "from": self.name,
                        "match_index": 0,
                    },
                    daemon=True,
                )
                result_events.append(resp)
                return result_events

        # Append new entries
        for entry_dict in entries:
            idx = entry_dict["index"]
            entry_term = entry_dict["term"]
            existing = self._log.get(idx)

            if existing and existing.term != entry_term:
                self._log.truncate_from(idx)
                self._log.append(entry_term, entry_dict["command"])
            elif not existing:
                self._log.append(entry_term, entry_dict["command"])

        # Update commit index
        if leader_commit > self._log.commit_index:
            new_commit = min(leader_commit, self._log.last_index)
            newly_committed = self._log.advance_commit(new_commit)
            self._apply_committed(newly_committed)

        resp = self._network.send(
            source=self,
            destination=sender,
            event_type="RaftAppendEntriesResponse",
            payload={
                "term": self._current_term,
                "success": True,
                "from": self.name,
                "match_index": self._log.last_index,
            },
            daemon=True,
        )
        result_events.append(resp)
        return result_events

    def _handle_append_entries_response(self, event: Event) -> list[Event]:
        metadata = event.context.get("metadata", {})
        term = metadata["term"]
        success = metadata["success"]
        follower = metadata.get("from")
        match_index = metadata.get("match_index", 0)

        if term > self._current_term:
            self._step_down(term)
            return [self._schedule_election_timeout()]

        if self._state != RaftState.LEADER:
            return []

        if follower is None:
            return []

        if success:
            self._next_index[follower] = match_index + 1
            self._match_index[follower] = match_index
            # Try to advance commit index
            return self._try_advance_commit()
        # Decrement next_index and retry
        current = self._next_index.get(follower, 1)
        self._next_index[follower] = max(1, current - 1)
        # Resend AppendEntries to this follower
        peer = self._find_peer(follower)
        if peer:
            prev_log_index = self._next_index[follower] - 1
            prev_log_term = 0
            if prev_log_index > 0:
                prev_entry = self._log.get(prev_log_index)
                if prev_entry:
                    prev_log_term = prev_entry.term

            entries = self._log.entries_after(prev_log_index)
            entry_dicts = [
                {"index": e.index, "term": e.term, "command": e.command} for e in entries
            ]

            msg = self._network.send(
                source=self,
                destination=peer,
                event_type="RaftAppendEntries",
                payload={
                    "term": self._current_term,
                    "leader_id": self.name,
                    "prev_log_index": prev_log_index,
                    "prev_log_term": prev_log_term,
                    "entries": entry_dicts,
                    "leader_commit": self._log.commit_index,
                },
                daemon=True,
            )
            return [msg]
        return []

    def _try_advance_commit(self) -> list[Event]:
        # Find the highest N such that a majority has match_index >= N
        # and log[N].term == current_term
        for n in range(self._log.last_index, self._log.commit_index, -1):
            entry = self._log.get(n)
            if entry is None or entry.term != self._current_term:
                continue

            count = 1  # self
            for match_idx in self._match_index.values():
                if match_idx >= n:
                    count += 1

            if count >= self.quorum_size:
                newly_committed = self._log.advance_commit(n)
                self._apply_committed(newly_committed)
                break
        return []

    def _apply_committed(self, entries: list[LogEntry]) -> None:
        for entry in entries:
            if entry.index > self._last_applied:
                result = self._state_machine.apply(entry.command)
                self._last_applied = entry.index
                self._commands_committed += 1

                future = self._pending_futures.pop(entry.index, None)
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
    def stats(self) -> RaftStats:
        return RaftStats(
            state=self._state,
            current_term=self._current_term,
            current_leader=self._leader,
            log_length=self._log.last_index,
            commit_index=self._log.commit_index,
            commands_committed=self._commands_committed,
            elections_started=self._elections_started,
            votes_received=self._total_votes_received,
        )

    def __repr__(self) -> str:
        return (
            f"RaftNode({self.name}, state={self._state.name}, "
            f"term={self._current_term}, leader={self._leader})"
        )
