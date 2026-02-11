"""Tests for FlexiblePaxosNode (asymmetric quorum Paxos)."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.core.temporal import Instant
from happysimulator.components.consensus.flexible_paxos import (
    FlexiblePaxosNode,
    FlexiblePaxosStats,
)
from happysimulator.components.consensus.paxos import Ballot
from happysimulator.components.network import Network


def make_clock(t=0.0):
    return Clock(Instant.from_seconds(t))


def make_flex_cluster(n=5, phase1_quorum=None, phase2_quorum=None):
    """Create n FlexiblePaxosNodes wired to a shared clock and network.

    When quorums are not specified, majority quorums for the target cluster
    size are computed upfront so the constructor and set_peers validation pass.
    """
    clock = make_clock(0.0)
    network = Network(name="test-net")
    network.set_clock(clock)

    # Compute default majority quorums for the target cluster size
    majority = (n // 2) + 1
    q1 = phase1_quorum if phase1_quorum is not None else majority
    q2 = phase2_quorum if phase2_quorum is not None else majority

    nodes = []
    for i in range(n):
        node = FlexiblePaxosNode(
            name=f"node-{i}",
            network=network,
            phase1_quorum=q1,
            phase2_quorum=q2,
        )
        node.set_clock(clock)
        nodes.append(node)

    for node in nodes:
        node.set_peers(nodes)

    return nodes, network, clock


class TestFlexiblePaxosDefaultQuorums:
    """Tests for default quorum calculation."""

    def test_default_quorums_majority(self):
        """Default quorums are both majority: (N // 2) + 1."""
        nodes, _, _ = make_flex_cluster(5)
        node = nodes[0]

        assert node.phase1_quorum == 3
        assert node.phase2_quorum == 3


class TestFlexiblePaxosCustomQuorums:
    """Tests for custom quorum configuration."""

    def test_custom_quorums_valid(self):
        """Custom quorums satisfying Q1 + Q2 > N are accepted."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        # 5 nodes, Q1=4, Q2=2 -> 4+2=6 > 5: valid
        node = FlexiblePaxosNode(
            name="node-0",
            network=network,
            peers=[],
            phase1_quorum=4,
            phase2_quorum=2,
        )

        assert node.phase1_quorum == 4
        assert node.phase2_quorum == 2

    def test_custom_quorums_invalid_raises(self):
        """Quorums violating Q1 + Q2 > N raise ValueError."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        # 1 node (self only), Q1=0, Q2=0 -> 0+0=0 <= 1: invalid
        with pytest.raises(ValueError, match="Q1 \\+ Q2 > N"):
            FlexiblePaxosNode(
                name="node-0",
                network=network,
                peers=[],
                phase1_quorum=0,
                phase2_quorum=0,
            )


class TestFlexiblePaxosAsymmetricFastWrites:
    """Tests for asymmetric quorum configuration favoring fast writes."""

    def test_asymmetric_fast_writes(self):
        """Small Q2 allows faster writes with fewer acks needed."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        # 5 nodes, Q1=4, Q2=2 -> fast writes (only 2 acks needed for accept)
        node = FlexiblePaxosNode(
            name="node-0",
            network=network,
            peers=[],
            phase1_quorum=4,
            phase2_quorum=2,
        )

        assert node.phase2_quorum == 2
        assert node.phase1_quorum == 4


class TestFlexiblePaxosAsymmetricFastRecovery:
    """Tests for asymmetric quorum configuration favoring fast recovery."""

    def test_asymmetric_fast_recovery(self):
        """Small Q1 allows faster leader recovery."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        # 5 nodes, Q1=2, Q2=4 -> fast recovery (only 2 promises needed)
        node = FlexiblePaxosNode(
            name="node-0",
            network=network,
            peers=[],
            phase1_quorum=2,
            phase2_quorum=4,
        )

        assert node.phase1_quorum == 2
        assert node.phase2_quorum == 4


class TestFlexiblePaxosInitialState:
    """Tests for initial state."""

    def test_initial_state(self):
        """Fresh FlexiblePaxosNode is not leader with empty log."""
        nodes, _, _ = make_flex_cluster(3)
        node = nodes[0]

        assert node.is_leader is False
        assert node.leader is None
        assert node.log.last_index == 0
        assert node.log.commit_index == 0


class TestFlexiblePaxosSubmit:
    """Tests for command submission."""

    def test_submit_queues_if_not_leader(self):
        """Submitting when not leader queues the command."""
        nodes, _, _ = make_flex_cluster(3)
        node = nodes[0]

        future = node.submit({"op": "set", "key": "k", "value": "v"})

        assert isinstance(future, SimFuture)
        assert len(node._pending_commands) == 1


class TestFlexiblePaxosPrepareHandler:
    """Tests for handling Prepare messages."""

    def test_prepare_handler(self):
        """Acceptor promises to a Prepare with a sufficiently high ballot."""
        nodes, _, clock = make_flex_cluster(3)
        node = nodes[1]

        prepare_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="FlexPaxosPrepare",
            target=node,
            context={"metadata": {
                "ballot_number": 5,
                "ballot_node": "node-0",
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = node.handle_event(prepare_event)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].event_type == "FlexPaxosPromise"


class TestFlexiblePaxosAcceptHandler:
    """Tests for handling Accept messages."""

    def test_accept_handler(self):
        """Follower accepts a valid Accept message."""
        nodes, _, clock = make_flex_cluster(3)
        follower = nodes[1]

        accept_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="FlexPaxosAccept",
            target=follower,
            context={"metadata": {
                "ballot_number": 1,
                "ballot_node": "node-0",
                "slot": 1,
                "command": {"op": "set", "key": "k", "value": "v"},
                "commit_index": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = follower.handle_event(accept_event)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].event_type == "FlexPaxosAccepted"
        assert follower.log.last_index == 1


class TestFlexiblePaxosNackHandler:
    """Tests for handling Nack messages."""

    def test_nack_handler_updates_ballot(self):
        """Handling a nack with a higher ballot updates the current ballot."""
        nodes, _, clock = make_flex_cluster(3)
        node = nodes[0]

        nack_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="FlexPaxosNack",
            target=node,
            context={"metadata": {
                "ballot_number": 99,
                "ballot_node": "node-2",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        node.handle_event(nack_event)

        assert node._current_ballot.number == 99
        assert node.is_leader is False


class TestFlexiblePaxosStats:
    """Tests for FlexiblePaxosStats."""

    def test_stats(self):
        """stats returns a FlexiblePaxosStats with correct fields."""
        nodes, _, _ = make_flex_cluster(3)
        node = nodes[0]

        stats = node.stats

        assert isinstance(stats, FlexiblePaxosStats)
        assert stats.is_leader is False
        assert stats.log_length == 0
        assert stats.commit_index == 0
        assert stats.commands_committed == 0
        assert stats.phase1_quorum == node.phase1_quorum
        assert stats.phase2_quorum == node.phase2_quorum


class TestFlexiblePaxosRepr:
    """Tests for __repr__."""

    def test_repr(self):
        """repr includes name, leader status, and quorum sizes."""
        nodes, _, _ = make_flex_cluster(3)
        node = nodes[0]

        r = repr(node)

        assert "FlexiblePaxosNode" in r
        assert "node-0" in r
        assert "Q1=" in r
        assert "Q2=" in r
