"""Tests for MembershipProtocol (SWIM-style failure detection)."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.components.consensus.membership import (
    MembershipProtocol,
    MemberState,
    MemberInfo,
    MembershipStats,
)
from happysimulator.components.network import Network


def make_clock(t=0.0):
    return Clock(Instant.from_seconds(t))


class DummyEntity(Entity):
    """Minimal entity for testing membership peer references."""

    def handle_event(self, event):
        return None


def make_membership(name="mp-1", clock=None, probe_interval=1.0, suspicion_timeout=5.0):
    """Create a MembershipProtocol wired to a clock and network."""
    clock = clock or make_clock(0.0)
    network = Network(name="test-net")
    network.set_clock(clock)
    mp = MembershipProtocol(
        name=name,
        network=network,
        probe_interval=probe_interval,
        suspicion_timeout=suspicion_timeout,
    )
    mp.set_clock(clock)
    return mp, network, clock


class TestMembershipAddMember:
    """Tests for adding members to the protocol."""

    def test_add_member(self):
        """Adding a member makes it appear in alive_members."""
        mp, _, _ = make_membership()
        peer = DummyEntity(name="peer-1")

        mp.add_member(peer)

        assert "peer-1" in mp.alive_members

    def test_add_self_ignored(self):
        """Adding self is a no-op."""
        mp, _, _ = make_membership(name="self-node")
        self_entity = DummyEntity(name="self-node")

        mp.add_member(self_entity)

        assert len(mp.alive_members) == 0


class TestMembershipInitialState:
    """Tests for initial member states."""

    def test_initial_state(self):
        """All added members start in ALIVE state."""
        mp, _, _ = make_membership()
        for i in range(3):
            mp.add_member(DummyEntity(name=f"peer-{i}"))

        assert len(mp.alive_members) == 3
        assert len(mp.suspected_members) == 0
        assert len(mp.dead_members) == 0


class TestMembershipStats:
    """Tests for MembershipStats snapshot."""

    def test_stats(self):
        """Stats reflect current membership state and counters."""
        mp, _, _ = make_membership()
        mp.add_member(DummyEntity(name="peer-1"))
        mp.add_member(DummyEntity(name="peer-2"))

        stats = mp.stats

        assert isinstance(stats, MembershipStats)
        assert stats.alive_count == 2
        assert stats.suspect_count == 0
        assert stats.dead_count == 0
        assert stats.probes_sent == 0
        assert stats.indirect_probes_sent == 0
        assert stats.acks_received == 0
        assert stats.updates_disseminated == 0


class TestMemberStateQuery:
    """Tests for querying individual member states."""

    def test_member_state_query(self):
        """get_member_state returns the correct state for known members."""
        mp, _, _ = make_membership()
        mp.add_member(DummyEntity(name="peer-1"))

        assert mp.get_member_state("peer-1") == MemberState.ALIVE
        assert mp.get_member_state("unknown") is None


class TestMembershipProbeTick:
    """Tests for probe tick event generation."""

    def test_probe_tick_generates_events(self):
        """start() + handling probe tick generates ping, timeout, and next tick."""
        mp, network, clock = make_membership()
        peer = DummyEntity(name="peer-1")
        peer.set_clock(clock)
        mp.add_member(peer)

        # start() should schedule a MembershipProbeTick
        start_events = mp.start()
        assert len(start_events) == 1
        assert start_events[0].event_type == "MembershipProbeTick"

        # Advance clock to tick time and handle
        tick_time = start_events[0].time
        clock.update(tick_time)
        result = mp.handle_event(start_events[0])

        # Should contain: ping event, indirect-ping timeout, next probe tick
        assert isinstance(result, list)
        event_types = [e.event_type for e in result]
        # At minimum we expect a ping to network, a timeout, and next tick
        assert "MembershipProbeTick" in event_types
        assert mp.stats.probes_sent >= 1


class TestMembershipPing:
    """Tests for ping/ack message handling."""

    def test_ping_generates_ack(self):
        """Handling a ping generates an ack response via network."""
        mp, network, clock = make_membership()
        sender = DummyEntity(name="peer-1")
        sender.set_clock(clock)
        mp.add_member(sender)

        ping_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="MembershipPing",
            target=mp,
            context={"metadata": {"from": "peer-1", "incarnation": 0, "updates": []}},
        )
        clock.update(Instant.from_seconds(1.0))
        result = mp.handle_event(ping_event)

        assert isinstance(result, list)
        assert len(result) == 1
        # The ack is sent via network.send, so it targets the network
        assert result[0].event_type == "MembershipAck"

    def test_ack_records_heartbeat(self):
        """Handling an ack increments acks_received counter."""
        mp, network, clock = make_membership()
        sender = DummyEntity(name="peer-1")
        sender.set_clock(clock)
        mp.add_member(sender)

        ack_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="MembershipAck",
            target=mp,
            context={"metadata": {"from": "peer-1", "incarnation": 0, "updates": []}},
        )
        clock.update(Instant.from_seconds(1.0))
        mp.handle_event(ack_event)

        assert mp.stats.acks_received == 1


class TestMembershipSuspicionTimeout:
    """Tests for suspicion and dead detection."""

    def test_suspicion_timeout_marks_dead(self):
        """A suspected member is declared dead on suspicion timeout."""
        mp, network, clock = make_membership()
        peer = DummyEntity(name="peer-1")
        peer.set_clock(clock)
        mp.add_member(peer)

        # Manually set member to SUSPECT state
        info = mp._members["peer-1"]
        info.state = MemberState.SUSPECT

        # Create and handle suspicion timeout
        timeout_event = Event(
            time=Instant.from_seconds(6.0),
            event_type="MembershipSuspicionTimeout",
            target=mp,
            context={"metadata": {"suspect": "peer-1"}},
        )
        clock.update(Instant.from_seconds(6.0))
        mp.handle_event(timeout_event)

        assert mp.get_member_state("peer-1") == MemberState.DEAD
        assert "peer-1" in mp.dead_members


class TestMembershipApplyUpdates:
    """Tests for piggybacked membership update application."""

    def test_apply_updates_suspect(self):
        """Applying a 'suspect' update transitions an ALIVE member to SUSPECT."""
        mp, _, _ = make_membership()
        peer = DummyEntity(name="peer-1")
        mp.add_member(peer)

        mp._apply_updates([{"member": "peer-1", "state": "suspect", "incarnation": 0}])

        assert mp.get_member_state("peer-1") == MemberState.SUSPECT

    def test_apply_updates_dead(self):
        """Applying a 'dead' update transitions a member to DEAD."""
        mp, _, _ = make_membership()
        peer = DummyEntity(name="peer-1")
        mp.add_member(peer)

        mp._apply_updates([{"member": "peer-1", "state": "dead", "incarnation": 0}])

        assert mp.get_member_state("peer-1") == MemberState.DEAD

    def test_incarnation_overrides_stale(self):
        """A higher incarnation 'alive' update overrides a stale suspicion."""
        mp, _, _ = make_membership()
        peer = DummyEntity(name="peer-1")
        mp.add_member(peer)

        # First, suspect with incarnation 0
        mp._apply_updates([{"member": "peer-1", "state": "suspect", "incarnation": 0}])
        assert mp.get_member_state("peer-1") == MemberState.SUSPECT

        # Then, alive with higher incarnation
        mp._apply_updates([{"member": "peer-1", "state": "alive", "incarnation": 1}])
        assert mp.get_member_state("peer-1") == MemberState.ALIVE


class TestMembershipListQueries:
    """Tests for alive/suspected/dead list properties."""

    def test_alive_suspected_dead_lists(self):
        """Alive, suspected, and dead lists reflect current states."""
        mp, _, _ = make_membership()
        mp.add_member(DummyEntity(name="a"))
        mp.add_member(DummyEntity(name="b"))
        mp.add_member(DummyEntity(name="c"))

        # Leave "a" alive, suspect "b", kill "c"
        mp._members["b"].state = MemberState.SUSPECT
        mp._members["c"].state = MemberState.DEAD

        assert mp.alive_members == ["a"]
        assert mp.suspected_members == ["b"]
        assert mp.dead_members == ["c"]


class TestMembershipRepr:
    """Tests for __repr__."""

    def test_repr(self):
        """repr includes name and member counts."""
        mp, _, _ = make_membership(name="cluster")
        mp.add_member(DummyEntity(name="peer-1"))

        r = repr(mp)

        assert "MembershipProtocol" in r
        assert "cluster" in r
        assert "alive=1" in r
