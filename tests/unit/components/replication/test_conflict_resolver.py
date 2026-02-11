"""Tests for ConflictResolver implementations."""

import pytest

from happysimulator.components.replication.conflict_resolver import (
    ConflictResolver,
    CustomResolver,
    LastWriterWins,
    VectorClockMerge,
    VersionedValue,
)
from happysimulator.core.logical_clocks import HLCTimestamp


class TestVersionedValue:
    """Tests for VersionedValue dataclass."""

    def test_create_with_float_timestamp(self):
        """VersionedValue works with float timestamp."""
        v = VersionedValue(value="hello", timestamp=1.0, writer_id="node-1")
        assert v.value == "hello"
        assert v.timestamp == 1.0
        assert v.writer_id == "node-1"
        assert v.vector_clock is None

    def test_create_with_hlc_timestamp(self):
        """VersionedValue works with HLCTimestamp."""
        ts = HLCTimestamp(physical_ns=1000, logical=0, node_id="node-1")
        v = VersionedValue(value="hello", timestamp=ts, writer_id="node-1")
        assert v.timestamp == ts

    def test_create_with_vector_clock(self):
        """VersionedValue accepts optional vector clock."""
        v = VersionedValue(
            value="hello",
            timestamp=1.0,
            writer_id="node-1",
            vector_clock={"node-1": 3, "node-2": 1},
        )
        assert v.vector_clock == {"node-1": 3, "node-2": 1}

    def test_frozen(self):
        """VersionedValue is frozen."""
        v = VersionedValue(value="hello", timestamp=1.0, writer_id="node-1")
        with pytest.raises(AttributeError):
            v.value = "world"


class TestLastWriterWins:
    """Tests for LastWriterWins resolver."""

    def test_higher_timestamp_wins(self):
        """Version with higher float timestamp wins."""
        resolver = LastWriterWins()
        v1 = VersionedValue(value="old", timestamp=1.0, writer_id="node-1")
        v2 = VersionedValue(value="new", timestamp=2.0, writer_id="node-2")

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "new"

    def test_order_independent(self):
        """Result is the same regardless of input order."""
        resolver = LastWriterWins()
        v1 = VersionedValue(value="old", timestamp=1.0, writer_id="node-1")
        v2 = VersionedValue(value="new", timestamp=2.0, writer_id="node-2")

        assert resolver.resolve("key", [v1, v2]).value == "new"
        assert resolver.resolve("key", [v2, v1]).value == "new"

    def test_tiebreak_by_writer_id(self):
        """Same timestamp breaks tie by writer_id."""
        resolver = LastWriterWins()
        v1 = VersionedValue(value="from-a", timestamp=1.0, writer_id="a")
        v2 = VersionedValue(value="from-b", timestamp=1.0, writer_id="b")

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "from-b"  # "b" > "a"

    def test_hlc_timestamp(self):
        """HLC timestamps compare correctly."""
        resolver = LastWriterWins()
        ts1 = HLCTimestamp(physical_ns=1000, logical=0, node_id="node-1")
        ts2 = HLCTimestamp(physical_ns=2000, logical=0, node_id="node-2")
        v1 = VersionedValue(value="old", timestamp=ts1, writer_id="node-1")
        v2 = VersionedValue(value="new", timestamp=ts2, writer_id="node-2")

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "new"

    def test_hlc_logical_tiebreak(self):
        """HLC timestamps with same physical break tie by logical."""
        resolver = LastWriterWins()
        ts1 = HLCTimestamp(physical_ns=1000, logical=0, node_id="node-1")
        ts2 = HLCTimestamp(physical_ns=1000, logical=1, node_id="node-2")
        v1 = VersionedValue(value="old", timestamp=ts1, writer_id="node-1")
        v2 = VersionedValue(value="new", timestamp=ts2, writer_id="node-2")

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "new"

    def test_single_version(self):
        """Single version is returned as-is."""
        resolver = LastWriterWins()
        v = VersionedValue(value="only", timestamp=1.0, writer_id="node-1")

        winner = resolver.resolve("key", [v])

        assert winner is v

    def test_three_versions(self):
        """Works with more than two versions."""
        resolver = LastWriterWins()
        versions = [
            VersionedValue(value="a", timestamp=1.0, writer_id="n1"),
            VersionedValue(value="c", timestamp=3.0, writer_id="n3"),
            VersionedValue(value="b", timestamp=2.0, writer_id="n2"),
        ]

        winner = resolver.resolve("key", versions)

        assert winner.value == "c"

    def test_implements_protocol(self):
        """LastWriterWins satisfies ConflictResolver protocol."""
        assert isinstance(LastWriterWins(), ConflictResolver)


class TestVectorClockMerge:
    """Tests for VectorClockMerge resolver."""

    def test_causal_dominance(self):
        """Causally dominant version wins."""
        resolver = VectorClockMerge()
        v1 = VersionedValue(
            value="old", timestamp=1.0, writer_id="n1",
            vector_clock={"n1": 1, "n2": 0},
        )
        v2 = VersionedValue(
            value="new", timestamp=2.0, writer_id="n2",
            vector_clock={"n1": 1, "n2": 1},
        )

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "new"

    def test_causal_dominance_reversed(self):
        """Causally dominant version wins regardless of list order."""
        resolver = VectorClockMerge()
        v1 = VersionedValue(
            value="old", timestamp=1.0, writer_id="n1",
            vector_clock={"n1": 1, "n2": 0},
        )
        v2 = VersionedValue(
            value="new", timestamp=2.0, writer_id="n2",
            vector_clock={"n1": 1, "n2": 1},
        )

        winner = resolver.resolve("key", [v2, v1])

        assert winner.value == "new"

    def test_concurrent_falls_back_to_lww(self):
        """Concurrent versions fall back to LWW when no merge_fn."""
        resolver = VectorClockMerge()
        v1 = VersionedValue(
            value="from-1", timestamp=1.0, writer_id="n1",
            vector_clock={"n1": 1, "n2": 0},
        )
        v2 = VersionedValue(
            value="from-2", timestamp=2.0, writer_id="n2",
            vector_clock={"n1": 0, "n2": 1},
        )

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "from-2"  # Higher timestamp

    def test_concurrent_uses_merge_fn(self):
        """Concurrent versions use merge_fn when provided."""
        def merge(key, a, b):
            return VersionedValue(
                value=f"{a.value}+{b.value}",
                timestamp=max(a.timestamp, b.timestamp),
                writer_id="merged",
            )

        resolver = VectorClockMerge(merge_fn=merge)
        v1 = VersionedValue(
            value="from-1", timestamp=1.0, writer_id="n1",
            vector_clock={"n1": 1, "n2": 0},
        )
        v2 = VersionedValue(
            value="from-2", timestamp=2.0, writer_id="n2",
            vector_clock={"n1": 0, "n2": 1},
        )

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "from-1+from-2"

    def test_missing_vector_clock_treated_as_empty(self):
        """None vector clock treated as empty (all zeros)."""
        resolver = VectorClockMerge()
        v1 = VersionedValue(value="a", timestamp=1.0, writer_id="n1")
        v2 = VersionedValue(
            value="b", timestamp=2.0, writer_id="n2",
            vector_clock={"n2": 1},
        )

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "b"  # v2 dominates (has non-zero clock)

    def test_implements_protocol(self):
        """VectorClockMerge satisfies ConflictResolver protocol."""
        assert isinstance(VectorClockMerge(), ConflictResolver)


class TestCustomResolver:
    """Tests for CustomResolver."""

    def test_delegates_to_function(self):
        """CustomResolver delegates to the user function."""
        def pick_first(key, versions):
            return versions[0]

        resolver = CustomResolver(pick_first)
        v1 = VersionedValue(value="first", timestamp=1.0, writer_id="n1")
        v2 = VersionedValue(value="second", timestamp=2.0, writer_id="n2")

        winner = resolver.resolve("key", [v1, v2])

        assert winner.value == "first"

    def test_receives_key(self):
        """CustomResolver passes key to the function."""
        captured_key = None

        def capture_key(key, versions):
            nonlocal captured_key
            captured_key = key
            return versions[0]

        resolver = CustomResolver(capture_key)
        v = VersionedValue(value="x", timestamp=1.0, writer_id="n1")
        resolver.resolve("user:123", [v])

        assert captured_key == "user:123"

    def test_implements_protocol(self):
        """CustomResolver satisfies ConflictResolver protocol."""
        resolver = CustomResolver(lambda k, vs: vs[0])
        assert isinstance(resolver, ConflictResolver)
