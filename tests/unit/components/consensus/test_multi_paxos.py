"""Tests for MultiPaxosNode (multi-decree Paxos with stable leader)."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.core.temporal import Instant
from happysimulator.components.consensus.multi_paxos import MultiPaxosNode, MultiPaxosStats
from happysimulator.components.consensus.paxos import Ballot
from happysimulator.components.network import Network


def make_clock(t=0.0):
    return Clock(Instant.from_seconds(t))


def make_multi_paxos_cluster(n=3):
    """Create n MultiPaxosNodes wired to a shared clock and network."""
    clock = make_clock(0.0)
    network = Network(name="test-net")
    network.set_clock(clock)

    nodes = []
    for i in range(n):
        node = MultiPaxosNode(name=f"node-{i}", network=network)
        node.set_clock(clock)
        nodes.append(node)

    for node in nodes:
        node.set_peers(nodes)

    return nodes, network, clock


class TestMultiPaxosInitialState:
    """Tests for initial state."""

    def test_initial_state(self):
        """Fresh MultiPaxosNode has no leader and empty log."""
        nodes, _, _ = make_multi_paxos_cluster(3)
        node = nodes[0]

        assert node.is_leader is False
        assert node.leader is None
        assert node.log.last_index == 0
        assert node.log.commit_index == 0


class TestMultiPaxosQuorum:
    """Tests for quorum size calculation."""

    def test_quorum_size(self):
        """Quorum is (N // 2) + 1."""
        nodes, _, _ = make_multi_paxos_cluster(5)
        assert nodes[0].quorum_size == 3

        nodes_3, _, _ = make_multi_paxos_cluster(3)
        assert nodes_3[0].quorum_size == 2


class TestMultiPaxosSetPeers:
    """Tests for set_peers method."""

    def test_set_peers(self):
        """set_peers excludes self from the peer list."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        node = MultiPaxosNode(name="node-0", network=network)
        node.set_clock(clock)
        peer1 = MultiPaxosNode(name="node-1", network=network)
        peer2 = MultiPaxosNode(name="node-2", network=network)
        peer1.set_clock(clock)
        peer2.set_clock(clock)

        node.set_peers([node, peer1, peer2])

        assert len(node._peers) == 2


class TestMultiPaxosSubmit:
    """Tests for command submission."""

    def test_submit_as_non_leader_queues(self):
        """Submitting when not leader queues the command."""
        nodes, _, _ = make_multi_paxos_cluster(3)
        node = nodes[0]

        future = node.submit({"op": "set", "key": "k", "value": "v"})

        assert isinstance(future, SimFuture)
        assert len(node._pending_commands) == 1


class TestMultiPaxosPhase1:
    """Tests for Phase 1 (Prepare)."""

    def test_begin_phase1_sends_prepare(self):
        """_begin_phase1 sends Prepare to all peers."""
        nodes, _, clock = make_multi_paxos_cluster(3)
        node = nodes[0]

        events = node._begin_phase1()

        # Should send Prepare to each peer (2 peers) + potentially self-become-leader events
        prepare_events = [e for e in events if e.event_type == "MultiPaxosPrepare"]
        assert len(prepare_events) == 2


class TestMultiPaxosPrepareHandler:
    """Tests for handling Prepare messages."""

    def test_prepare_handler_promises(self):
        """Acceptor promises to a Prepare with a sufficiently high ballot."""
        nodes, _, clock = make_multi_paxos_cluster(3)
        node = nodes[1]

        prepare_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="MultiPaxosPrepare",
            target=node,
            context={"metadata": {
                "ballot_number": 5,
                "ballot_node": "node-0",
                "log_length": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = node.handle_event(prepare_event)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].event_type == "MultiPaxosPromise"

    def test_prepare_handler_nacks(self):
        """Acceptor nacks Prepare with lower ballot than its own."""
        nodes, _, clock = make_multi_paxos_cluster(3)
        node = nodes[1]

        # Set a high ballot on this node
        node._current_ballot = Ballot(10, "node-1")

        prepare_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="MultiPaxosPrepare",
            target=node,
            context={"metadata": {
                "ballot_number": 5,
                "ballot_node": "node-0",
                "log_length": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = node.handle_event(prepare_event)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].event_type == "MultiPaxosNack"


class TestMultiPaxosBecomeLeader:
    """Tests for leader establishment."""

    def test_become_leader_processes_pending(self):
        """When becoming leader, pending commands are assigned to slots."""
        nodes, _, clock = make_multi_paxos_cluster(3)
        node = nodes[0]

        # Queue a pending command
        future = SimFuture()
        node._pending_commands.append(({"op": "set", "key": "k", "value": "v"}, future))

        node._become_leader()

        assert node.is_leader is True
        assert node.leader == "node-0"
        assert len(node._pending_commands) == 0
        assert node.log.last_index >= 1


class TestMultiPaxosAcceptHandler:
    """Tests for handling Accept messages."""

    def test_accept_handler(self):
        """Follower accepts a valid Accept message and appends to log."""
        nodes, _, clock = make_multi_paxos_cluster(3)
        follower = nodes[1]

        accept_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="MultiPaxosAccept",
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
        assert result[0].event_type == "MultiPaxosAccepted"
        assert follower.log.last_index == 1


class TestMultiPaxosCommit:
    """Tests for commit and state machine application."""

    def test_commit_applies_to_state_machine(self):
        """Committed entries are applied to the state machine."""
        nodes, _, clock = make_multi_paxos_cluster(3)
        node = nodes[0]

        # Manually become leader and add an entry
        node._is_leader = True
        node._leader = "node-0"
        node._current_ballot = Ballot(1, "node-0")
        node.log.append(1, {"op": "set", "key": "k", "value": "v"})

        # Commit it
        committed = node.log.advance_commit(1)
        node._apply_committed(committed)

        assert node._commands_committed == 1
        assert node._state_machine.data.get("k") == "v"


class TestMultiPaxosStats:
    """Tests for MultiPaxosStats."""

    def test_stats(self):
        """stats returns a MultiPaxosStats with correct fields."""
        nodes, _, _ = make_multi_paxos_cluster(3)
        node = nodes[0]

        stats = node.stats

        assert isinstance(stats, MultiPaxosStats)
        assert stats.is_leader is False
        assert stats.log_length == 0
        assert stats.commit_index == 0
        assert stats.commands_committed == 0


class TestMultiPaxosRepr:
    """Tests for __repr__."""

    def test_repr(self):
        """repr includes name, leader status, and ballot."""
        nodes, _, _ = make_multi_paxos_cluster(3)
        node = nodes[0]

        r = repr(node)

        assert "MultiPaxosNode" in r
        assert "node-0" in r
        assert "leader=" in r
        assert "ballot=" in r
