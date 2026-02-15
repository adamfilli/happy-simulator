"""Tests for PaxosNode (single-decree Paxos)."""

from happysimulator.components.consensus.paxos import Ballot, PaxosNode, PaxosStats
from happysimulator.components.network import Network
from happysimulator.core.clock import Clock
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.core.temporal import Instant


def make_clock(t=0.0):
    return Clock(Instant.from_seconds(t))


def make_paxos_cluster(n=3):
    """Create n PaxosNodes wired to a shared clock and network."""
    clock = make_clock(0.0)
    network = Network(name="test-net")
    network.set_clock(clock)

    nodes = []
    for i in range(n):
        node = PaxosNode(name=f"node-{i}", network=network)
        node.set_clock(clock)
        nodes.append(node)

    # Wire peers
    for node in nodes:
        node.set_peers(nodes)

    return nodes, network, clock


class TestBallotOrdering:
    """Tests for Ballot comparison semantics."""

    def test_ballot_ordering(self):
        """Ballots are ordered by number first."""
        b1 = Ballot(1, "a")
        b2 = Ballot(2, "a")

        assert b1 < b2
        assert not b2 < b1

    def test_ballot_tiebreak_by_node_id(self):
        """Ballots with equal number are ordered by node_id."""
        b1 = Ballot(1, "a")
        b2 = Ballot(1, "b")

        assert b1 < b2
        assert not b2 < b1


class TestPaxosInitialState:
    """Tests for PaxosNode initial state."""

    def test_initial_state_not_decided(self):
        """A fresh PaxosNode has not decided."""
        nodes, _, _ = make_paxos_cluster(3)
        node = nodes[0]

        assert node.is_decided is False
        assert node.decided_value is None


class TestPaxosQuorum:
    """Tests for quorum size calculation."""

    def test_quorum_size(self):
        """Quorum is (N // 2) + 1."""
        # 5 nodes: quorum = 3
        nodes_5, _, _ = make_paxos_cluster(5)
        assert nodes_5[0].quorum_size == 3

        # 3 nodes: quorum = 2
        nodes_3, _, _ = make_paxos_cluster(3)
        assert nodes_3[0].quorum_size == 2

        # 1 node: quorum = 1
        nodes_1, _, _ = make_paxos_cluster(1)
        assert nodes_1[0].quorum_size == 1


class TestPaxosSetPeers:
    """Tests for set_peers method."""

    def test_set_peers(self):
        """set_peers excludes self from the peer list."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        node = PaxosNode(name="node-0", network=network)
        node.set_clock(clock)
        peer1 = PaxosNode(name="node-1", network=network)
        peer2 = PaxosNode(name="node-2", network=network)
        peer1.set_clock(clock)
        peer2.set_clock(clock)

        node.set_peers([node, peer1, peer2])

        assert len(node._peers) == 2
        peer_names = {p.name for p in node._peers}
        assert "node-0" not in peer_names


class TestPaxosPropose:
    """Tests for the propose method."""

    def test_propose_returns_future(self):
        """propose() returns a SimFuture."""
        nodes, _, _ = make_paxos_cluster(3)
        node = nodes[0]

        future = node.propose("value-1")

        assert isinstance(future, SimFuture)
        assert node._proposals_started == 1

    def test_propose_when_already_decided(self):
        """propose() returns pre-resolved future when already decided."""
        nodes, _, _ = make_paxos_cluster(3)
        node = nodes[0]
        node._decided = True
        node._decided_value = "decided-value"

        future = node.propose("new-value")

        assert future.is_resolved is True
        assert future.value == "decided-value"


class TestPaxosSelfPromise:
    """Tests for Phase 1 self-promise."""

    def test_self_promise_in_phase1(self):
        """start_phase1() includes a self-promise in phase1 responses."""
        nodes, _, _clock = make_paxos_cluster(3)
        node = nodes[0]
        node.propose("val")

        node.start_phase1()

        # Check self-promise was recorded
        ballot_num = node._current_ballot.number
        assert len(node._phase1_responses[ballot_num]) >= 1
        assert node._promises_received >= 1


class TestPaxosPrepareHandler:
    """Tests for handling Prepare messages."""

    def test_prepare_handler_promises(self):
        """An acceptor promises to a Prepare with a sufficiently high ballot."""
        nodes, _, clock = make_paxos_cluster(3)
        node = nodes[1]  # acceptor

        prepare_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="PaxosPrepare",
            target=node,
            context={
                "metadata": {
                    "ballot_number": 5,
                    "ballot_node": "node-0",
                    "source": "node-0",
                }
            },
        )
        clock.update(Instant.from_seconds(1.0))
        result = node.handle_event(prepare_event)

        assert isinstance(result, list)
        assert len(result) == 1
        # The response should be a Promise sent via network
        assert result[0].event_type == "PaxosPromise"

    def test_prepare_handler_nacks_lower_ballot(self):
        """An acceptor nacks a Prepare with a lower ballot than promised."""
        nodes, _, clock = make_paxos_cluster(3)
        node = nodes[1]

        # First, promise a high ballot
        node._promised_ballot = Ballot(10, "node-2")

        prepare_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="PaxosPrepare",
            target=node,
            context={
                "metadata": {
                    "ballot_number": 5,
                    "ballot_node": "node-0",
                    "source": "node-0",
                }
            },
        )
        clock.update(Instant.from_seconds(1.0))
        result = node.handle_event(prepare_event)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].event_type == "PaxosNack"


class TestPaxosAcceptHandler:
    """Tests for handling Accept messages."""

    def test_accept_handler_accepts(self):
        """An acceptor accepts a value for a ballot it has promised."""
        nodes, _, clock = make_paxos_cluster(3)
        node = nodes[1]

        accept_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="PaxosAccept",
            target=node,
            context={
                "metadata": {
                    "ballot_number": 5,
                    "ballot_node": "node-0",
                    "value": "hello",
                    "source": "node-0",
                }
            },
        )
        clock.update(Instant.from_seconds(1.0))
        result = node.handle_event(accept_event)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].event_type == "PaxosAccepted"
        assert node._accepted_value == "hello"

    def test_accept_handler_nacks_lower_ballot(self):
        """An acceptor nacks Accept with a ballot lower than its promised ballot."""
        nodes, _, clock = make_paxos_cluster(3)
        node = nodes[1]

        # Promise a high ballot first
        node._promised_ballot = Ballot(10, "node-2")

        accept_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="PaxosAccept",
            target=node,
            context={
                "metadata": {
                    "ballot_number": 5,
                    "ballot_node": "node-0",
                    "value": "hello",
                    "source": "node-0",
                }
            },
        )
        clock.update(Instant.from_seconds(1.0))
        result = node.handle_event(accept_event)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].event_type == "PaxosNack"


class TestPaxosDecided:
    """Tests for learning decided values."""

    def test_decided_handler_learns_value(self):
        """Handling a PaxosDecided message sets the decided value."""
        nodes, _, clock = make_paxos_cluster(3)
        node = nodes[1]

        decided_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="PaxosDecided",
            target=node,
            context={"metadata": {"value": "final-answer"}},
        )
        clock.update(Instant.from_seconds(1.0))
        node.handle_event(decided_event)

        assert node.is_decided is True
        assert node.decided_value == "final-answer"


class TestPaxosStats:
    """Tests for PaxosStats."""

    def test_stats(self):
        """stats returns a PaxosStats with initial zeros."""
        nodes, _, _ = make_paxos_cluster(3)
        node = nodes[0]

        stats = node.stats

        assert isinstance(stats, PaxosStats)
        assert stats.proposals_started == 0
        assert stats.proposals_succeeded == 0
        assert stats.proposals_failed == 0
        assert stats.promises_received == 0
        assert stats.nacks_received == 0
        assert stats.accepts_received == 0
        assert stats.decided_value is None


class TestPaxosRepr:
    """Tests for __repr__."""

    def test_repr(self):
        """repr includes name, decided status, and ballot."""
        nodes, _, _ = make_paxos_cluster(3)
        node = nodes[0]

        r = repr(node)

        assert "PaxosNode" in r
        assert "node-0" in r
        assert "decided=" in r
        assert "ballot=" in r
