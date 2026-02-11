"""Tests for PNCounter CRDT."""

import pytest

from happysimulator.components.crdt.pn_counter import PNCounter
from happysimulator.components.crdt.protocol import CRDT


class TestPNCounterCreation:
    """Tests for PNCounter construction."""

    def test_creates_with_node_id(self):
        c = PNCounter("node-a")
        assert c.node_id == "node-a"

    def test_initial_value_is_zero(self):
        c = PNCounter("node-a")
        assert c.value == 0

    def test_implements_crdt_protocol(self):
        c = PNCounter("node-a")
        assert isinstance(c, CRDT)

    def test_repr(self):
        c = PNCounter("node-a")
        assert "node-a" in repr(c)


class TestPNCounterIncrementDecrement:
    """Tests for increment and decrement operations."""

    def test_increment(self):
        c = PNCounter("node-a")
        c.increment(5)
        assert c.value == 5

    def test_decrement(self):
        c = PNCounter("node-a")
        c.increment(10)
        c.decrement(3)
        assert c.value == 7

    def test_value_can_be_negative(self):
        c = PNCounter("node-a")
        c.decrement(5)
        assert c.value == -5

    def test_increments_property(self):
        c = PNCounter("node-a")
        c.increment(10)
        c.decrement(3)
        assert c.increments == 10

    def test_decrements_property(self):
        c = PNCounter("node-a")
        c.increment(10)
        c.decrement(3)
        assert c.decrements == 3

    def test_multiple_operations(self):
        c = PNCounter("node-a")
        c.increment(10)
        c.decrement(3)
        c.increment(2)
        c.decrement(1)
        assert c.value == 8  # 12 - 4


class TestPNCounterMerge:
    """Tests for merge operations."""

    def test_merge_two_nodes(self):
        a = PNCounter("node-a")
        b = PNCounter("node-b")
        a.increment(5)
        b.increment(3)
        b.decrement(1)

        a.merge(b)
        assert a.value == 7  # 5 + 3 - 1

    def test_merge_preserves_both_p_and_n(self):
        a = PNCounter("node-a")
        b = PNCounter("node-b")
        a.increment(10)
        a.decrement(2)
        b.increment(5)
        b.decrement(3)

        a.merge(b)
        assert a.increments == 15  # 10 + 5
        assert a.decrements == 5   # 2 + 3
        assert a.value == 10       # 15 - 5

    def test_merge_is_idempotent(self):
        a = PNCounter("node-a")
        b = PNCounter("node-b")
        a.increment(5)
        b.decrement(2)

        a.merge(b)
        a.merge(b)
        assert a.value == 3

    def test_merge_is_commutative(self):
        a = PNCounter("node-a")
        b = PNCounter("node-b")
        a.increment(5)
        b.decrement(2)

        a_copy = PNCounter.from_dict(a.to_dict())
        b_copy = PNCounter.from_dict(b.to_dict())

        a_copy.merge(b)
        b_copy.merge(a)
        assert a_copy.value == b_copy.value


class TestPNCounterSerialization:
    """Tests for serialization."""

    def test_round_trip(self):
        c = PNCounter("node-a")
        c.increment(10)
        c.decrement(3)
        d = c.to_dict()
        c2 = PNCounter.from_dict(d)
        assert c == c2
        assert c2.value == 7

    def test_to_dict_structure(self):
        c = PNCounter("node-a")
        c.increment(5)
        d = c.to_dict()
        assert d["type"] == "PNCounter"
        assert d["node_id"] == "node-a"
        assert "p" in d
        assert "n" in d

    def test_equality(self):
        a = PNCounter("node-a")
        b = PNCounter("node-a")
        assert a == b
        a.increment()
        assert a != b
