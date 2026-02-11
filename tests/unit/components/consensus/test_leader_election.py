"""Tests for LeaderElection entity."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.components.consensus.leader_election import (
    LeaderElection,
    ElectionStats,
)
from happysimulator.components.consensus.election_strategies import BullyStrategy
from happysimulator.components.network import Network


def make_clock(t=0.0):
    return Clock(Instant.from_seconds(t))


class DummyEntity(Entity):
    """Minimal entity for testing."""

    def handle_event(self, event):
        return None


def make_election(name="le-1", members=None, strategy=None, clock=None):
    """Create a LeaderElection wired to a clock and network."""
    clock = clock or make_clock(0.0)
    network = Network(name="test-net")
    network.set_clock(clock)
    le = LeaderElection(
        name=name,
        network=network,
        members=members or {},
        strategy=strategy,
    )
    le.set_clock(clock)
    return le, network, clock


class TestLeaderElectionInitialState:
    """Tests for initial leader election state."""

    def test_initial_state_no_leader(self):
        """No leader is set initially."""
        le, _, _ = make_election()

        assert le.current_leader is None
        assert le.current_term == 0
        assert le.is_leader is False


class TestLeaderElectionStart:
    """Tests for starting the election process."""

    def test_start_schedules_timeout(self):
        """start() schedules an ElectionTimeoutCheck event."""
        le, _, _ = make_election()

        events = le.start()

        assert len(events) == 1
        assert events[0].event_type == "ElectionTimeoutCheck"


class TestLeaderElectionVictory:
    """Tests for victory and heartbeat handling."""

    def test_victory_sets_leader(self):
        """Handling an ElectionVictory message sets the leader."""
        le, network, clock = make_election()

        victory_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="ElectionVictory",
            target=le,
            context={"metadata": {"leader": "node-5", "term": 1}},
        )
        clock.update(Instant.from_seconds(1.0))
        le.handle_event(victory_event)

        assert le.current_leader == "node-5"

    def test_heartbeat_updates_leader(self):
        """Handling a LeaderHeartbeat updates the known leader."""
        le, network, clock = make_election()

        hb_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="LeaderHeartbeat",
            target=le,
            context={"metadata": {"leader": "node-3", "term": 2}},
        )
        clock.update(Instant.from_seconds(1.0))
        le.handle_event(hb_event)

        assert le.current_leader == "node-3"
        assert le.current_term == 2

    def test_heartbeat_resets_timeout(self):
        """A heartbeat updates the last heartbeat time."""
        le, network, clock = make_election()

        hb_event = Event(
            time=Instant.from_seconds(5.0),
            event_type="LeaderHeartbeat",
            target=le,
            context={"metadata": {"leader": "node-3", "term": 1}},
        )
        clock.update(Instant.from_seconds(5.0))
        le.handle_event(hb_event)

        assert le._last_leader_heartbeat == 5.0


class TestLeaderElectionElection:
    """Tests for election mechanics."""

    def test_election_increments_term(self):
        """Starting an election increments the term."""
        le, network, clock = make_election()
        peer = DummyEntity(name="peer-1")
        peer.set_clock(clock)
        le.add_member(peer)

        initial_term = le.current_term
        le._start_election()

        assert le.current_term == initial_term + 1


class TestLeaderElectionStats:
    """Tests for election statistics."""

    def test_stats(self):
        """stats returns an ElectionStats dataclass with correct fields."""
        le, _, _ = make_election()

        stats = le.stats

        assert isinstance(stats, ElectionStats)
        assert stats.current_leader is None
        assert stats.current_term == 0
        assert stats.elections_started == 0
        assert stats.elections_won == 0
        assert stats.elections_participated == 0


class TestLeaderElectionIsLeader:
    """Tests for is_leader property."""

    def test_is_leader(self):
        """is_leader is True when current_leader matches this node's name."""
        le, _, _ = make_election(name="leader-node")
        le._current_leader = "leader-node"

        assert le.is_leader is True

        le._current_leader = "other-node"
        assert le.is_leader is False


class TestLeaderElectionAddMember:
    """Tests for adding members."""

    def test_add_member(self):
        """add_member registers a member for election participation."""
        le, _, _ = make_election()
        peer = DummyEntity(name="peer-1")

        le.add_member(peer)

        assert "peer-1" in le._members


class TestLeaderElectionBully:
    """Tests for bully strategy integration."""

    def test_bully_highest_wins(self):
        """With BullyStrategy, the highest-named node with no higher peers wins."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        # "node-z" is highest, so if it starts an election it should
        # declare victory immediately (no higher peers)
        le = LeaderElection(
            name="node-z",
            network=network,
            strategy=BullyStrategy(),
        )
        le.set_clock(clock)

        # Add a lower member
        peer = DummyEntity(name="node-a")
        peer.set_clock(clock)
        le.add_member(peer)

        events = le._start_election()

        # Should become leader since no higher nodes
        assert le.current_leader == "node-z"
        assert le.is_leader is True


class TestLeaderElectionRepr:
    """Tests for __repr__."""

    def test_repr(self):
        """repr includes name, leader, and term."""
        le, _, _ = make_election(name="le-1")

        r = repr(le)

        assert "LeaderElection" in r
        assert "le-1" in r
        assert "leader=" in r
        assert "term=" in r
