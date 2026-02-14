"""Tests for GCounter CRDT."""

import pytest

from happysimulator.components.crdt.g_counter import GCounter
from happysimulator.components.crdt.protocol import CRDT


class TestGCounterCreation:
    """Tests for GCounter construction."""

    def test_creates_with_node_id(self):
        c = GCounter("node-a")
        assert c.node_id == "node-a"

    def test_initial_value_is_zero(self):
        c = GCounter("node-a")
        assert c.value == 0

    def test_implements_crdt_protocol(self):
        c = GCounter("node-a")
        assert isinstance(c, CRDT)

    def test_repr(self):
        c = GCounter("node-a")
        assert "node-a" in repr(c)
        assert "0" in repr(c)


class TestGCounterIncrement:
    """Tests for increment operations."""

    def test_single_increment(self):
        c = GCounter("node-a")
        c.increment()
        assert c.value == 1

    def test_multiple_increments(self):
        c = GCounter("node-a")
        c.increment()
        c.increment()
        c.increment()
        assert c.value == 3

    def test_increment_by_n(self):
        c = GCounter("node-a")
        c.increment(5)
        assert c.value == 5

    def test_increment_negative_raises(self):
        c = GCounter("node-a")
        with pytest.raises(ValueError, match="positive"):
            c.increment(0)

    def test_node_value(self):
        c = GCounter("node-a")
        c.increment(3)
        assert c.node_value("node-a") == 3
        assert c.node_value("node-b") == 0


class TestGCounterMerge:
    """Tests for merge operations."""

    def test_merge_two_nodes(self):
        a = GCounter("node-a")
        b = GCounter("node-b")
        a.increment(5)
        b.increment(3)

        a.merge(b)
        assert a.value == 8

    def test_merge_is_idempotent(self):
        a = GCounter("node-a")
        b = GCounter("node-b")
        a.increment(5)
        b.increment(3)

        a.merge(b)
        a.merge(b)
        assert a.value == 8

    def test_merge_is_commutative(self):
        a = GCounter("node-a")
        b = GCounter("node-b")
        a.increment(5)
        b.increment(3)

        a_copy = GCounter.from_dict(a.to_dict())
        b_copy = GCounter.from_dict(b.to_dict())

        a_copy.merge(b)
        b_copy.merge(a)
        assert a_copy.value == b_copy.value

    def test_merge_is_associative(self):
        a = GCounter("node-a")
        b = GCounter("node-b")
        c = GCounter("node-c")
        a.increment(1)
        b.increment(2)
        c.increment(3)

        # (a merge b) merge c
        ab = GCounter.from_dict(a.to_dict())
        ab.merge(b)
        ab.merge(c)

        # a merge (b merge c)
        bc = GCounter.from_dict(b.to_dict())
        bc.merge(c)
        a2 = GCounter.from_dict(a.to_dict())
        a2.merge(bc)

        assert ab.value == a2.value == 6

    def test_merge_takes_max_per_node(self):
        """After divergent increments, merge keeps max per node."""
        a = GCounter("node-a")
        b = GCounter("node-a")  # Same node_id, simulating replicas

        a.increment(5)
        b.increment(3)

        a.merge(b)
        assert a.value == 5  # max(5, 3) = 5


class TestGCounterSerialization:
    """Tests for serialization."""

    def test_round_trip(self):
        c = GCounter("node-a")
        c.increment(7)
        d = c.to_dict()
        c2 = GCounter.from_dict(d)
        assert c == c2
        assert c2.value == 7

    def test_to_dict_structure(self):
        c = GCounter("node-a")
        c.increment(3)
        d = c.to_dict()
        assert d["type"] == "GCounter"
        assert d["node_id"] == "node-a"
        assert d["counts"] == {"node-a": 3}

    def test_equality(self):
        a = GCounter("node-a")
        b = GCounter("node-a")
        assert a == b
        a.increment()
        assert a != b
