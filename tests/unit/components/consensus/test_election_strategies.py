"""Tests for election strategies: Bully, Ring, and Randomized."""

import pytest
import random

from happysimulator.components.consensus.election_strategies import (
    ElectionStrategy,
    BullyStrategy,
    RingStrategy,
    RandomizedStrategy,
)


class TestBullyStrategy:
    """Tests for BullyStrategy."""

    def test_bully_should_start_election(self):
        """Bully strategy always recommends starting an election."""
        strategy = BullyStrategy()

        assert strategy.should_start_election("node-a", ["node-a", "node-b"]) is True

    def test_bully_no_higher_sends_victory(self):
        """When no higher-ID nodes exist, victory messages are generated."""
        strategy = BullyStrategy()

        messages = strategy.get_election_messages(
            node_id="node-z",
            alive_members=["node-a", "node-b", "node-z"],
            term=1,
        )

        assert len(messages) == 2  # victory to node-a and node-b
        for msg in messages:
            assert msg["event_type"] == "ElectionVictory"
            assert msg["payload"]["leader"] == "node-z"

    def test_bully_higher_sends_challenge(self):
        """When higher-ID nodes exist, challenge messages are sent to them."""
        strategy = BullyStrategy()

        messages = strategy.get_election_messages(
            node_id="node-a",
            alive_members=["node-a", "node-b", "node-c"],
            term=1,
        )

        # node-b and node-c are higher than node-a
        assert len(messages) == 2
        targets = {msg["target"] for msg in messages}
        assert targets == {"node-b", "node-c"}
        for msg in messages:
            assert msg["event_type"] == "ElectionChallenge"

    def test_bully_challenge_from_lower_suppresses(self):
        """Higher node receiving challenge sends suppress and starts own election."""
        strategy = BullyStrategy()

        result = strategy.handle_election_message(
            node_id="node-c",
            message_type="ElectionChallenge",
            payload={"challenger": "node-a", "term": 1},
            alive_members=["node-a", "node-b", "node-c"],
        )

        assert len(result["response_messages"]) == 1
        assert result["response_messages"][0]["event_type"] == "ElectionSuppress"
        assert result["start_own_election"] is True
        assert result["leader"] is None

    def test_bully_victory_sets_leader(self):
        """Handling victory message sets the leader."""
        strategy = BullyStrategy()

        result = strategy.handle_election_message(
            node_id="node-a",
            message_type="ElectionVictory",
            payload={"leader": "node-c", "term": 1},
            alive_members=["node-a", "node-b", "node-c"],
        )

        assert result["leader"] == "node-c"
        assert result["suppress_election"] is True


class TestRingStrategy:
    """Tests for RingStrategy."""

    def test_ring_should_start_election(self):
        """Ring strategy always recommends starting an election."""
        strategy = RingStrategy()

        assert strategy.should_start_election("node-a", ["node-a", "node-b"]) is True

    def test_ring_get_messages_sends_token(self):
        """get_election_messages sends a token to the next node in the ring."""
        strategy = RingStrategy()

        messages = strategy.get_election_messages(
            node_id="node-a",
            alive_members=["node-a", "node-b", "node-c"],
            term=1,
        )

        assert len(messages) == 1
        assert messages[0]["event_type"] == "ElectionToken"
        assert messages[0]["payload"]["initiator"] == "node-a"
        assert "node-a" in messages[0]["payload"]["candidates"]

    def test_ring_token_returns_to_initiator(self):
        """When the token returns to the initiator, the highest candidate wins."""
        strategy = RingStrategy()

        result = strategy.handle_election_message(
            node_id="node-a",
            message_type="ElectionToken",
            payload={
                "initiator": "node-a",
                "candidates": ["node-a", "node-b", "node-c"],
                "term": 1,
            },
            alive_members=["node-a", "node-b", "node-c"],
        )

        assert result["leader"] == "node-c"  # max of candidates
        assert result["suppress_election"] is True
        # Victory messages sent to other members
        assert len(result["response_messages"]) == 2

    def test_ring_forward_token(self):
        """Non-initiator adds itself and forwards the token."""
        strategy = RingStrategy()

        result = strategy.handle_election_message(
            node_id="node-b",
            message_type="ElectionToken",
            payload={
                "initiator": "node-a",
                "candidates": ["node-a"],
                "term": 1,
            },
            alive_members=["node-a", "node-b", "node-c"],
        )

        assert result["leader"] is None
        assert len(result["response_messages"]) == 1
        forwarded = result["response_messages"][0]
        assert forwarded["event_type"] == "ElectionToken"
        assert "node-b" in forwarded["payload"]["candidates"]
        assert "node-a" in forwarded["payload"]["candidates"]

    def test_ring_victory_sets_leader(self):
        """Handling victory from ring sets the leader."""
        strategy = RingStrategy()

        result = strategy.handle_election_message(
            node_id="node-b",
            message_type="ElectionVictory",
            payload={"leader": "node-c", "term": 1},
            alive_members=["node-a", "node-b", "node-c"],
        )

        assert result["leader"] == "node-c"
        assert result["suppress_election"] is True


class TestRandomizedStrategy:
    """Tests for RandomizedStrategy."""

    def test_randomized_should_start(self):
        """Randomized strategy always recommends starting an election."""
        strategy = RandomizedStrategy()

        assert strategy.should_start_election("node-a", ["node-a", "node-b"]) is True

    def test_randomized_sends_ballot(self):
        """get_election_messages sends ballots to all other members."""
        random.seed(42)
        strategy = RandomizedStrategy()

        messages = strategy.get_election_messages(
            node_id="node-a",
            alive_members=["node-a", "node-b", "node-c"],
            term=1,
        )

        assert len(messages) == 2  # to node-b and node-c
        targets = {msg["target"] for msg in messages}
        assert targets == {"node-b", "node-c"}
        for msg in messages:
            assert msg["event_type"] == "ElectionBallot"
            assert "ballot" in msg["payload"]
            assert msg["payload"]["ballot"] > 0

    def test_randomized_handles_ballot(self):
        """Receiving a ballot sends a response with own ballot."""
        random.seed(42)
        strategy = RandomizedStrategy()

        result = strategy.handle_election_message(
            node_id="node-b",
            message_type="ElectionBallot",
            payload={"from": "node-a", "ballot": 500, "term": 1},
            alive_members=["node-a", "node-b"],
        )

        assert len(result["response_messages"]) == 1
        resp = result["response_messages"][0]
        assert resp["target"] == "node-a"
        assert resp["event_type"] == "ElectionBallotResponse"
        assert resp["payload"]["ballot"] > 0


class TestProtocolCompliance:
    """Tests for ElectionStrategy protocol compliance."""

    def test_protocol_compliance(self):
        """All strategies implement the ElectionStrategy protocol."""
        assert isinstance(BullyStrategy(), ElectionStrategy)
        assert isinstance(RingStrategy(), ElectionStrategy)
        assert isinstance(RandomizedStrategy(), ElectionStrategy)
