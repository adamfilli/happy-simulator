"""Tests for ORSet CRDT."""

from happysimulator.components.crdt.or_set import ORSet
from happysimulator.components.crdt.protocol import CRDT


class TestORSetCreation:
    """Tests for ORSet construction."""

    def test_creates_with_node_id(self):
        s = ORSet("node-a")
        assert s.node_id == "node-a"

    def test_initial_set_is_empty(self):
        s = ORSet("node-a")
        assert len(s) == 0
        assert s.elements == frozenset()

    def test_implements_crdt_protocol(self):
        s = ORSet("node-a")
        assert isinstance(s, CRDT)

    def test_repr(self):
        s = ORSet("node-a")
        assert "node-a" in repr(s)


class TestORSetAddRemove:
    """Tests for add and remove operations."""

    def test_add_element(self):
        s = ORSet("node-a")
        s.add("apple")
        assert "apple" in s

    def test_add_multiple_elements(self):
        s = ORSet("node-a")
        s.add("apple")
        s.add("banana")
        assert s.elements == frozenset({"apple", "banana"})

    def test_remove_element(self):
        s = ORSet("node-a")
        s.add("apple")
        s.remove("apple")
        assert "apple" not in s

    def test_remove_nonexistent_is_noop(self):
        s = ORSet("node-a")
        s.remove("apple")  # no error
        assert len(s) == 0

    def test_add_after_remove(self):
        s = ORSet("node-a")
        s.add("apple")
        s.remove("apple")
        s.add("apple")
        assert "apple" in s

    def test_add_same_element_twice(self):
        s = ORSet("node-a")
        s.add("apple")
        s.add("apple")
        assert len(s) == 1
        assert "apple" in s


class TestORSetMerge:
    """Tests for merge operations."""

    def test_merge_disjoint_sets(self):
        a = ORSet("node-a")
        b = ORSet("node-b")
        a.add("apple")
        b.add("banana")

        a.merge(b)
        assert a.elements == frozenset({"apple", "banana"})

    def test_merge_overlapping_adds(self):
        a = ORSet("node-a")
        b = ORSet("node-b")
        a.add("apple")
        b.add("apple")

        a.merge(b)
        assert "apple" in a
        assert len(a) == 1

    def test_add_wins_over_concurrent_remove(self):
        """Concurrent add on one node and remove on another: add wins."""
        a = ORSet("node-a")
        b = ORSet("node-b")

        # Both start with apple
        a.add("apple")
        b.merge(a)  # b now has apple too

        # Concurrent: a removes, b re-adds
        a.remove("apple")
        b.add("apple")  # new tag on node-b

        # Merge: b's new tag survives a's remove
        a.merge(b)
        assert "apple" in a

    def test_remove_does_not_affect_unseen_adds(self):
        """Remove only affects tags observed at remove time."""
        a = ORSet("node-a")
        b = ORSet("node-b")

        a.add("x")
        # b doesn't know about x yet
        # b independently adds x
        b.add("x")

        # a removes x (only its own tag)
        a.remove("x")

        # merge: b's tag survives
        a.merge(b)
        assert "x" in a

    def test_merge_is_idempotent(self):
        a = ORSet("node-a")
        b = ORSet("node-b")
        a.add("apple")
        b.add("banana")

        a.merge(b)
        a.merge(b)
        assert a.elements == frozenset({"apple", "banana"})

    def test_merge_is_commutative(self):
        a = ORSet("node-a")
        b = ORSet("node-b")
        a.add("apple")
        b.add("banana")

        a_copy = ORSet.from_dict(a.to_dict())
        b_copy = ORSet.from_dict(b.to_dict())

        a_copy.merge(b)
        b_copy.merge(a)
        assert a_copy.elements == b_copy.elements

    def test_merge_three_nodes(self):
        a = ORSet("node-a")
        b = ORSet("node-b")
        c = ORSet("node-c")
        a.add("apple")
        b.add("banana")
        c.add("cherry")

        a.merge(b)
        a.merge(c)
        assert a.elements == frozenset({"apple", "banana", "cherry"})


class TestORSetContains:
    """Tests for containment checks."""

    def test_contains_true(self):
        s = ORSet("node-a")
        s.add("apple")
        assert s.contains("apple")
        assert "apple" in s

    def test_contains_false(self):
        s = ORSet("node-a")
        assert not s.contains("apple")
        assert "apple" not in s

    def test_contains_after_remove(self):
        s = ORSet("node-a")
        s.add("apple")
        s.remove("apple")
        assert not s.contains("apple")


class TestORSetIteration:
    """Tests for iteration and length."""

    def test_len(self):
        s = ORSet("node-a")
        s.add("apple")
        s.add("banana")
        assert len(s) == 2

    def test_iter(self):
        s = ORSet("node-a")
        s.add("apple")
        s.add("banana")
        assert set(s) == {"apple", "banana"}

    def test_value_property(self):
        s = ORSet("node-a")
        s.add("apple")
        assert s.value == frozenset({"apple"})


class TestORSetSerialization:
    """Tests for serialization."""

    def test_round_trip(self):
        s = ORSet("node-a")
        s.add("apple")
        s.add("banana")
        d = s.to_dict()
        s2 = ORSet.from_dict(d)
        assert s2.elements == frozenset({"apple", "banana"})

    def test_to_dict_structure(self):
        s = ORSet("node-a")
        s.add("apple")
        d = s.to_dict()
        assert d["type"] == "ORSet"
        assert d["node_id"] == "node-a"
        assert "entries" in d

    def test_equality(self):
        a = ORSet("node-a")
        b = ORSet("node-a")
        a.add("apple")
        b.add("apple")
        # Different tags (different sequence numbers from same node)
        # but same elements — equality checks active tag sets
        # These will NOT be equal because tags differ
        # (a has tag ("node-a", 0), b has tag ("node-a", 0))
        # Actually they will be equal since both are ("node-a", 0)
        assert a == b
