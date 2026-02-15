"""Tests for LWWRegister CRDT."""

from happysimulator.components.crdt.lww_register import LWWRegister
from happysimulator.components.crdt.protocol import CRDT
from happysimulator.core.logical_clocks import HLCTimestamp


def _ts(physical_ns: int, logical: int = 0, node_id: str = "node-a") -> HLCTimestamp:
    """Shorthand for creating test timestamps."""
    return HLCTimestamp(physical_ns=physical_ns, logical=logical, node_id=node_id)


class TestLWWRegisterCreation:
    """Tests for LWWRegister construction."""

    def test_creates_with_node_id(self):
        r = LWWRegister("node-a")
        assert r.node_id == "node-a"

    def test_initial_value_is_none(self):
        r = LWWRegister("node-a")
        assert r.value is None
        assert r.timestamp is None

    def test_creates_with_initial_value(self):
        ts = _ts(1000)
        r = LWWRegister("node-a", value="hello", timestamp=ts)
        assert r.value == "hello"
        assert r.timestamp == ts

    def test_implements_crdt_protocol(self):
        r = LWWRegister("node-a")
        assert isinstance(r, CRDT)

    def test_repr(self):
        r = LWWRegister("node-a", value="test")
        assert "node-a" in repr(r)
        assert "test" in repr(r)


class TestLWWRegisterSet:
    """Tests for set operations."""

    def test_set_value(self):
        r = LWWRegister("node-a")
        r.set("hello", _ts(1000))
        assert r.value == "hello"

    def test_newer_timestamp_wins(self):
        r = LWWRegister("node-a")
        r.set("first", _ts(1000))
        r.set("second", _ts(2000))
        assert r.value == "second"

    def test_older_timestamp_ignored(self):
        r = LWWRegister("node-a")
        r.set("first", _ts(2000))
        r.set("second", _ts(1000))
        assert r.value == "first"

    def test_get_is_alias_for_value(self):
        r = LWWRegister("node-a")
        r.set("hello", _ts(1000))
        assert r.get() == r.value


class TestLWWRegisterMerge:
    """Tests for merge operations."""

    def test_merge_higher_timestamp_wins(self):
        a = LWWRegister("node-a")
        b = LWWRegister("node-b")
        a.set("from-a", _ts(1000, node_id="node-a"))
        b.set("from-b", _ts(2000, node_id="node-b"))

        a.merge(b)
        assert a.value == "from-b"

    def test_merge_lower_timestamp_no_change(self):
        a = LWWRegister("node-a")
        b = LWWRegister("node-b")
        a.set("from-a", _ts(2000, node_id="node-a"))
        b.set("from-b", _ts(1000, node_id="node-b"))

        a.merge(b)
        assert a.value == "from-a"

    def test_merge_tie_breaks_on_node_id(self):
        """Same physical+logical, tie-break on node_id (lexicographic)."""
        a = LWWRegister("node-a")
        b = LWWRegister("node-b")
        a.set("from-a", _ts(1000, node_id="node-a"))
        b.set("from-b", _ts(1000, node_id="node-b"))

        # node-b > node-a lexicographically, so b wins
        a.merge(b)
        assert a.value == "from-b"

    def test_merge_with_unwritten_other_is_noop(self):
        a = LWWRegister("node-a")
        b = LWWRegister("node-b")
        a.set("hello", _ts(1000))

        a.merge(b)  # b has no value
        assert a.value == "hello"

    def test_merge_into_unwritten_register(self):
        a = LWWRegister("node-a")
        b = LWWRegister("node-b")
        b.set("from-b", _ts(1000))

        a.merge(b)
        assert a.value == "from-b"


class TestLWWRegisterSerialization:
    """Tests for serialization."""

    def test_round_trip(self):
        r = LWWRegister("node-a")
        r.set("hello", _ts(1000, logical=5))
        d = r.to_dict()
        r2 = LWWRegister.from_dict(d)
        assert r == r2
        assert r2.value == "hello"

    def test_round_trip_none_value(self):
        r = LWWRegister("node-a")
        d = r.to_dict()
        r2 = LWWRegister.from_dict(d)
        assert r2.value is None
        assert r2.timestamp is None

    def test_to_dict_structure(self):
        r = LWWRegister("node-a")
        r.set("test", _ts(1000))
        d = r.to_dict()
        assert d["type"] == "LWWRegister"
        assert d["node_id"] == "node-a"
        assert d["value"] == "test"
        assert d["timestamp"]["physical_ns"] == 1000
