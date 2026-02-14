"""Tests for RaftNode consensus protocol."""

import pytest
import random

from happysimulator.core.clock import Clock
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.core.temporal import Instant
from happysimulator.components.consensus.raft import RaftNode, RaftStats, RaftState
from happysimulator.components.network import Network


def make_clock(t=0.0):
    return Clock(Instant.from_seconds(t))


def make_raft_cluster(n=3):
    """Create n RaftNodes wired to a shared clock and network."""
    clock = make_clock(0.0)
    network = Network(name="test-net")
    network.set_clock(clock)

    nodes = []
    for i in range(n):
        node = RaftNode(name=f"node-{i}", network=network)
        node.set_clock(clock)
        nodes.append(node)

    for node in nodes:
        node.set_peers(nodes)

    return nodes, network, clock


class TestRaftInitialState:
    """Tests for Raft initial state."""

    def test_initial_state_follower(self):
        """A fresh RaftNode starts as a FOLLOWER."""
        nodes, _, _ = make_raft_cluster(3)
        node = nodes[0]

        assert node.state == RaftState.FOLLOWER
        assert node.current_term == 0
        assert node.current_leader is None
        assert node.is_leader is False


class TestRaftQuorum:
    """Tests for quorum size calculation."""

    def test_quorum_size(self):
        """Quorum is (N // 2) + 1."""
        nodes_5, _, _ = make_raft_cluster(5)
        assert nodes_5[0].quorum_size == 3

        nodes_3, _, _ = make_raft_cluster(3)
        assert nodes_3[0].quorum_size == 2


class TestRaftSetPeers:
    """Tests for set_peers method."""

    def test_set_peers(self):
        """set_peers excludes self from the peer list."""
        clock = make_clock(0.0)
        network = Network(name="test-net")
        network.set_clock(clock)

        node = RaftNode(name="node-0", network=network)
        node.set_clock(clock)
        peer1 = RaftNode(name="node-1", network=network)
        peer2 = RaftNode(name="node-2", network=network)
        peer1.set_clock(clock)
        peer2.set_clock(clock)

        node.set_peers([node, peer1, peer2])

        assert len(node._peers) == 2


class TestRaftStart:
    """Tests for starting the Raft node."""

    def test_start_schedules_election_timeout(self):
        """start() schedules a RaftElectionTimeout event."""
        random.seed(42)
        nodes, _, _ = make_raft_cluster(3)
        node = nodes[0]

        events = node.start()

        assert len(events) == 1
        assert events[0].event_type == "RaftElectionTimeout"


class TestRaftElectionTimeout:
    """Tests for election timeout handling."""

    def test_election_timeout_starts_election(self):
        """Election timeout triggers an election."""
        random.seed(42)
        nodes, _, clock = make_raft_cluster(3)
        node = nodes[0]

        start_events = node.start()
        timeout_event = start_events[0]

        # Advance clock to timeout
        clock.update(timeout_event.time)
        result = node.handle_event(timeout_event)

        assert node.state == RaftState.CANDIDATE
        assert node.current_term == 1


class TestRaftCandidateState:
    """Tests for candidate behavior."""

    def test_candidate_increments_term(self):
        """Starting an election increments the term."""
        nodes, _, clock = make_raft_cluster(3)
        node = nodes[0]

        node._start_election()

        assert node.current_term == 1
        assert node.state == RaftState.CANDIDATE

    def test_candidate_votes_for_self(self):
        """A candidate votes for itself."""
        nodes, _, clock = make_raft_cluster(3)
        node = nodes[0]

        node._start_election()

        assert node._voted_for == "node-0"
        assert "node-0" in node._votes_received_set


class TestRaftRequestVote:
    """Tests for vote request handling."""

    def test_request_vote_grants_vote(self):
        """A follower grants a vote to a candidate with valid credentials."""
        nodes, _, clock = make_raft_cluster(3)
        follower = nodes[1]

        vote_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="RaftRequestVote",
            target=follower,
            context={"metadata": {
                "term": 1,
                "candidate_id": "node-0",
                "last_log_index": 0,
                "last_log_term": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = follower.handle_event(vote_event)

        assert isinstance(result, list)
        # Should contain vote response + election timeout reset
        vote_responses = [e for e in result if e.event_type == "RaftVoteResponse"]
        assert len(vote_responses) == 1
        # Check the vote was granted via the metadata
        assert follower._voted_for == "node-0"

    def test_request_vote_rejects_lower_term(self):
        """A follower rejects votes from a lower term."""
        nodes, _, clock = make_raft_cluster(3)
        follower = nodes[1]
        follower._current_term = 5

        vote_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="RaftRequestVote",
            target=follower,
            context={"metadata": {
                "term": 3,
                "candidate_id": "node-0",
                "last_log_index": 0,
                "last_log_term": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = follower.handle_event(vote_event)

        # Vote should not be granted (follower's voted_for should remain None)
        assert follower._voted_for is None

    def test_request_vote_rejects_stale_log(self):
        """A follower rejects a candidate with a less up-to-date log."""
        nodes, _, clock = make_raft_cluster(3)
        follower = nodes[1]

        # Give follower a log entry at term 2
        follower._log.append(2, {"op": "set", "key": "k", "value": "v"})

        vote_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="RaftRequestVote",
            target=follower,
            context={"metadata": {
                "term": 3,
                "candidate_id": "node-0",
                "last_log_index": 0,
                "last_log_term": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = follower.handle_event(vote_event)

        # Candidate's log is stale (term 0 < term 2), so vote should be rejected
        assert follower._voted_for is None


class TestRaftStepDown:
    """Tests for stepping down on higher term."""

    def test_step_down_on_higher_term(self):
        """A node steps down when it sees a higher term."""
        nodes, _, clock = make_raft_cluster(3)
        node = nodes[0]
        node._state = RaftState.CANDIDATE
        node._current_term = 3

        node._step_down(5)

        assert node.state == RaftState.FOLLOWER
        assert node.current_term == 5
        assert node._voted_for is None


class TestRaftBecomeLeader:
    """Tests for becoming leader."""

    def test_become_leader_initializes_state(self):
        """Becoming leader initializes next_index and match_index for peers."""
        nodes, _, clock = make_raft_cluster(3)
        node = nodes[0]
        node._current_term = 1
        node._state = RaftState.CANDIDATE

        node._become_leader()

        assert node.state == RaftState.LEADER
        assert node._leader == "node-0"
        assert node.is_leader is True
        # Should have next_index for each peer
        for peer in node._peers:
            assert peer.name in node._next_index
            assert peer.name in node._match_index


class TestRaftAppendEntries:
    """Tests for AppendEntries handling."""

    def test_append_entries_resets_timeout(self):
        """Receiving AppendEntries resets the election timeout."""
        nodes, _, clock = make_raft_cluster(3)
        follower = nodes[1]

        ae_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="RaftAppendEntries",
            target=follower,
            context={"metadata": {
                "term": 1,
                "leader_id": "node-0",
                "prev_log_index": 0,
                "prev_log_term": 0,
                "entries": [],
                "leader_commit": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = follower.handle_event(ae_event)

        assert isinstance(result, list)
        # Should contain election timeout reset
        timeout_events = [e for e in result if e.event_type == "RaftElectionTimeout"]
        assert len(timeout_events) == 1
        # Leader should be recognized
        assert follower._leader == "node-0"

    def test_append_entries_rejects_lower_term(self):
        """AppendEntries with lower term is rejected."""
        nodes, _, clock = make_raft_cluster(3)
        follower = nodes[1]
        follower._current_term = 5

        ae_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="RaftAppendEntries",
            target=follower,
            context={"metadata": {
                "term": 3,
                "leader_id": "node-0",
                "prev_log_index": 0,
                "prev_log_term": 0,
                "entries": [],
                "leader_commit": 0,
                "source": "node-0",
            }},
        )
        clock.update(Instant.from_seconds(1.0))
        result = follower.handle_event(ae_event)

        # Should contain a response with success=False
        response_events = [e for e in result if e.event_type == "RaftAppendEntriesResponse"]
        assert len(response_events) == 1


class TestRaftSubmit:
    """Tests for command submission."""

    def test_submit_as_leader_appends_to_log(self):
        """Submitting as leader appends the command to the log."""
        nodes, _, clock = make_raft_cluster(3)
        node = nodes[0]
        node._state = RaftState.LEADER
        node._current_term = 1
        node._leader = "node-0"

        future = node.submit({"op": "set", "key": "k", "value": "v"})

        assert isinstance(future, SimFuture)
        assert node.log.last_index == 1
        assert node.log.get(1).command == {"op": "set", "key": "k", "value": "v"}


class TestRaftStats:
    """Tests for RaftStats."""

    def test_stats(self):
        """stats returns a RaftStats with correct fields."""
        nodes, _, _ = make_raft_cluster(3)
        node = nodes[0]

        stats = node.stats

        assert isinstance(stats, RaftStats)
        assert stats.state == RaftState.FOLLOWER
        assert stats.current_term == 0
        assert stats.current_leader is None
        assert stats.log_length == 0
        assert stats.commit_index == 0
        assert stats.commands_committed == 0
        assert stats.elections_started == 0
        assert stats.votes_received == 0


class TestRaftRepr:
    """Tests for __repr__."""

    def test_repr(self):
        """repr includes name, state, term, and leader."""
        nodes, _, _ = make_raft_cluster(3)
        node = nodes[0]

        r = repr(node)

        assert "RaftNode" in r
        assert "node-0" in r
        assert "FOLLOWER" in r
        assert "term=" in r
        assert "leader=" in r
