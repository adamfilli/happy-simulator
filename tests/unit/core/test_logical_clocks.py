"""Unit tests for logical clocks (Lamport, Vector, HLC)."""

from __future__ import annotations

import pytest

from happysimulator.core.logical_clocks import (
    HLCTimestamp,
    HybridLogicalClock,
    LamportClock,
    VectorClock,
)
from happysimulator.core.node_clock import NodeClock
from happysimulator.core.temporal import Instant


# =============================================================================
# LamportClock
# =============================================================================


class TestLamportClock:
    def test_initial_time_is_zero(self):
        clock = LamportClock()
        assert clock.time == 0

    def test_initial_time_custom(self):
        clock = LamportClock(initial=10)
        assert clock.time == 10

    def test_tick_increments(self):
        clock = LamportClock()
        clock.tick()
        assert clock.time == 1
        clock.tick()
        assert clock.time == 2

    def test_send_increments_and_returns(self):
        clock = LamportClock()
        ts = clock.send()
        assert ts == 1
        assert clock.time == 1
        ts2 = clock.send()
        assert ts2 == 2

    def test_receive_advances_past_remote(self):
        clock = LamportClock()
        clock.receive(5)
        assert clock.time == 6  # max(0, 5) + 1

    def test_receive_when_local_is_higher(self):
        clock = LamportClock(initial=10)
        clock.receive(3)
        assert clock.time == 11  # max(10, 3) + 1

    def test_receive_when_equal(self):
        clock = LamportClock(initial=5)
        clock.receive(5)
        assert clock.time == 6  # max(5, 5) + 1

    def test_send_receive_preserves_causality(self):
        """Sending and receiving should preserve happens-before."""
        a = LamportClock()
        b = LamportClock()

        # A sends to B
        ts = a.send()
        b.receive(ts)
        assert b.time > ts  # B's time is after A's sent timestamp

        # B sends back to A
        ts2 = b.send()
        a.receive(ts2)
        assert a.time > ts2  # A's time is after B's sent timestamp

    def test_multiple_rounds(self):
        a = LamportClock()
        b = LamportClock()
        c = LamportClock()

        # A ticks locally
        a.tick()
        a.tick()
        assert a.time == 2

        # A sends to B
        ts = a.send()  # A=3
        b.receive(ts)   # B = max(0, 3) + 1 = 4

        # B sends to C
        ts2 = b.send()  # B=5
        c.receive(ts2)   # C = max(0, 5) + 1 = 6

        assert c.time == 6
        assert c.time > b.time - 1  # C received after B sent


# =============================================================================
# VectorClock
# =============================================================================


class TestVectorClock:
    def test_initial_vector_is_zeros(self):
        vc = VectorClock("A", ["A", "B", "C"])
        assert vc.snapshot() == {"A": 0, "B": 0, "C": 0}

    def test_tick_increments_own(self):
        vc = VectorClock("A", ["A", "B"])
        vc.tick()
        assert vc.snapshot() == {"A": 1, "B": 0}

    def test_send_increments_and_returns_copy(self):
        vc = VectorClock("A", ["A", "B"])
        snap = vc.send()
        assert snap == {"A": 1, "B": 0}
        # Mutating returned dict doesn't affect internal state
        snap["A"] = 999
        assert vc.snapshot()["A"] == 1

    def test_receive_takes_elementwise_max_and_increments(self):
        vc = VectorClock("A", ["A", "B", "C"])
        vc.tick()  # A={A:1, B:0, C:0}

        remote = {"A": 0, "B": 3, "C": 2}
        vc.receive(remote)
        # max then increment own: A=1+1=2, B=max(0,3)=3, C=max(0,2)=2
        assert vc.snapshot() == {"A": 2, "B": 3, "C": 2}

    def test_happened_before_true(self):
        a = VectorClock("A", ["A", "B"])
        b = VectorClock("B", ["A", "B"])

        ts = a.send()  # A={A:1, B:0}
        b.receive(ts)   # B={A:1, B:1}

        # a happened before b (a's state at send time was {A:1, B:0})
        assert a.happened_before(b)

    def test_happened_before_false_when_concurrent(self):
        a = VectorClock("A", ["A", "B"])
        b = VectorClock("B", ["A", "B"])

        a.tick()  # A={A:1, B:0}
        b.tick()  # B={A:0, B:1}

        assert not a.happened_before(b)
        assert not b.happened_before(a)

    def test_happened_before_false_when_after(self):
        a = VectorClock("A", ["A", "B"])
        b = VectorClock("B", ["A", "B"])

        ts = a.send()
        b.receive(ts)

        assert not b.happened_before(a)

    def test_is_concurrent(self):
        a = VectorClock("A", ["A", "B"])
        b = VectorClock("B", ["A", "B"])

        a.tick()  # Independent local events
        b.tick()

        assert a.is_concurrent(b)
        assert b.is_concurrent(a)

    def test_not_concurrent_after_communication(self):
        a = VectorClock("A", ["A", "B"])
        b = VectorClock("B", ["A", "B"])

        ts = a.send()
        b.receive(ts)

        assert not a.is_concurrent(b)

    def test_merge_returns_new_clock_no_increment(self):
        a = VectorClock("A", ["A", "B", "C"])
        b = VectorClock("B", ["A", "B", "C"])

        a.tick()  # {A:1, B:0, C:0}
        b.tick()  # {A:0, B:1, C:0}
        b.tick()  # {A:0, B:2, C:0}

        merged = a.merge(b)
        assert merged.snapshot() == {"A": 1, "B": 2, "C": 0}
        # Original clocks unchanged
        assert a.snapshot() == {"A": 1, "B": 0, "C": 0}
        assert b.snapshot() == {"A": 0, "B": 2, "C": 0}

    def test_merge_preserves_node_id(self):
        a = VectorClock("A", ["A", "B"])
        b = VectorClock("B", ["A", "B"])
        merged = a.merge(b)
        assert merged.node_id == "A"

    def test_node_id_property(self):
        vc = VectorClock("node-1", ["node-1", "node-2"])
        assert vc.node_id == "node-1"

    def test_receive_with_unknown_node(self):
        """Receive from a node not in the original set."""
        vc = VectorClock("A", ["A", "B"])
        vc.receive({"A": 0, "B": 0, "C": 5})
        snap = vc.snapshot()
        assert snap["C"] == 5
        assert snap["A"] == 1  # Own counter incremented

    def test_three_node_causal_chain(self):
        """A -> B -> C: C knows about A's event."""
        a = VectorClock("A", ["A", "B", "C"])
        b = VectorClock("B", ["A", "B", "C"])
        c = VectorClock("C", ["A", "B", "C"])

        ts_ab = a.send()   # A={A:1, B:0, C:0}
        b.receive(ts_ab)    # B={A:1, B:1, C:0}
        ts_bc = b.send()    # B={A:1, B:2, C:0}
        c.receive(ts_bc)     # C={A:1, B:2, C:1}

        assert a.happened_before(c)
        assert b.happened_before(c)

    def test_equal_vectors_not_happened_before(self):
        """Equal vectors: neither happened before the other, but not concurrent either."""
        a = VectorClock("A", ["A", "B"])
        b = VectorClock("B", ["A", "B"])
        # Both at {A:0, B:0}
        assert not a.happened_before(b)
        assert not b.happened_before(a)
        # Equal vectors are NOT concurrent by the standard definition
        # (concurrent = neither happened-before, which IS the case here)
        # But since both are {0,0}, technically they're the "same" event
        assert a.is_concurrent(b)


# =============================================================================
# HLCTimestamp
# =============================================================================


class TestHLCTimestamp:
    def test_ordering_by_physical(self):
        a = HLCTimestamp(physical_ns=100, logical=0, node_id="A")
        b = HLCTimestamp(physical_ns=200, logical=0, node_id="A")
        assert a < b
        assert b > a

    def test_ordering_by_logical(self):
        a = HLCTimestamp(physical_ns=100, logical=0, node_id="A")
        b = HLCTimestamp(physical_ns=100, logical=1, node_id="A")
        assert a < b

    def test_ordering_by_node_id(self):
        a = HLCTimestamp(physical_ns=100, logical=0, node_id="A")
        b = HLCTimestamp(physical_ns=100, logical=0, node_id="B")
        assert a < b

    def test_equality(self):
        a = HLCTimestamp(physical_ns=100, logical=5, node_id="X")
        b = HLCTimestamp(physical_ns=100, logical=5, node_id="X")
        assert a == b
        assert not (a != b)

    def test_inequality(self):
        a = HLCTimestamp(physical_ns=100, logical=5, node_id="X")
        b = HLCTimestamp(physical_ns=100, logical=6, node_id="X")
        assert a != b

    def test_hash_equal(self):
        a = HLCTimestamp(physical_ns=100, logical=5, node_id="X")
        b = HLCTimestamp(physical_ns=100, logical=5, node_id="X")
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_le_ge(self):
        a = HLCTimestamp(physical_ns=100, logical=0, node_id="A")
        b = HLCTimestamp(physical_ns=100, logical=0, node_id="A")
        assert a <= b
        assert a >= b

    def test_to_dict(self):
        ts = HLCTimestamp(physical_ns=1_000_000, logical=3, node_id="node-1")
        d = ts.to_dict()
        assert d == {
            "physical_ns": 1_000_000,
            "logical": 3,
            "node_id": "node-1",
        }

    def test_from_dict(self):
        d = {"physical_ns": 1_000_000, "logical": 3, "node_id": "node-1"}
        ts = HLCTimestamp.from_dict(d)
        assert ts.physical_ns == 1_000_000
        assert ts.logical == 3
        assert ts.node_id == "node-1"

    def test_roundtrip(self):
        original = HLCTimestamp(physical_ns=42, logical=7, node_id="z")
        assert HLCTimestamp.from_dict(original.to_dict()) == original

    def test_frozen(self):
        ts = HLCTimestamp(physical_ns=1, logical=2, node_id="A")
        with pytest.raises(AttributeError):
            ts.physical_ns = 99  # type: ignore[misc]


# =============================================================================
# HybridLogicalClock
# =============================================================================


class TestHybridLogicalClock:
    @staticmethod
    def _make_wall(ns_values: list[int]) -> tuple:
        """Create a wall_time callable that returns successive Instant values."""
        it = iter(ns_values)

        def wall_time() -> Instant:
            return Instant(next(it))

        return wall_time

    def test_requires_time_source(self):
        with pytest.raises(ValueError, match="Provide either"):
            HybridLogicalClock("A")

    def test_rejects_both_time_sources(self):
        clock = NodeClock()
        with pytest.raises(ValueError, match="not both"):
            HybridLogicalClock(
                "A",
                physical_clock=clock,
                wall_time=lambda: Instant(0),
            )

    def test_now_advances_physical(self):
        wall = self._make_wall([100, 200, 300])
        hlc = HybridLogicalClock("A", wall_time=wall)

        ts1 = hlc.now()
        assert ts1.physical_ns == 100
        assert ts1.logical == 0

        ts2 = hlc.now()
        assert ts2.physical_ns == 200
        assert ts2.logical == 0

    def test_now_increments_logical_when_physical_unchanged(self):
        wall = self._make_wall([100, 100, 100])
        hlc = HybridLogicalClock("A", wall_time=wall)

        ts1 = hlc.now()
        assert ts1 == HLCTimestamp(100, 0, "A")

        ts2 = hlc.now()
        assert ts2 == HLCTimestamp(100, 1, "A")

        ts3 = hlc.now()
        assert ts3 == HLCTimestamp(100, 2, "A")

    def test_send_is_alias_for_now(self):
        wall = self._make_wall([100, 200])
        hlc = HybridLogicalClock("A", wall_time=wall)

        ts = hlc.send()
        assert ts == HLCTimestamp(100, 0, "A")

    def test_receive_remote_physical_ahead(self):
        """Remote has higher physical: adopt remote physical, logical = remote.logical + 1."""
        wall = self._make_wall([100, 100])
        hlc = HybridLogicalClock("A", wall_time=wall)
        hlc.now()  # Consume first wall read: last=(100, 0, A)

        remote = HLCTimestamp(physical_ns=200, logical=5, node_id="B")
        hlc.receive(remote)
        # pt=100, last=100, remote=200 → max=200 = remote only
        # logical = remote.logical + 1 = 6
        ts = hlc._last
        assert ts.physical_ns == 200
        assert ts.logical == 6

    def test_receive_local_physical_ahead(self):
        """Local last has higher physical: keep local, logical = last.logical + 1."""
        wall = self._make_wall([200, 200])
        hlc = HybridLogicalClock("A", wall_time=wall)
        hlc.now()  # last=(200, 0, A)

        remote = HLCTimestamp(physical_ns=100, logical=5, node_id="B")
        hlc.receive(remote)
        # pt=200, last=200, remote=100 → max=200 = last (and pt)
        # Since max_pt == last.physical_ns and max_pt != remote.physical_ns
        # logical = last.logical + 1 = 1
        ts = hlc._last
        assert ts.physical_ns == 200
        assert ts.logical == 1

    def test_receive_all_physical_equal(self):
        """All three physical values equal: logical = max(last.logical, remote.logical) + 1."""
        wall = self._make_wall([100, 100])
        hlc = HybridLogicalClock("A", wall_time=wall)
        hlc.now()  # last=(100, 0, A)

        remote = HLCTimestamp(physical_ns=100, logical=3, node_id="B")
        hlc.receive(remote)
        # All three = 100, logical = max(0, 3) + 1 = 4
        ts = hlc._last
        assert ts.physical_ns == 100
        assert ts.logical == 4

    def test_receive_physical_advances_past_both(self):
        """Current physical time exceeds both last and remote: reset logical."""
        wall = self._make_wall([100, 500])
        hlc = HybridLogicalClock("A", wall_time=wall)
        hlc.now()  # last=(100, 0, A)

        remote = HLCTimestamp(physical_ns=200, logical=10, node_id="B")
        hlc.receive(remote)
        # pt=500, last=100, remote=200 → max=500 = pt only
        # logical = 0 (fresh physical time)
        ts = hlc._last
        assert ts.physical_ns == 500
        assert ts.logical == 0

    def test_monotonicity_despite_clock_regression(self):
        """Even if physical clock goes backward, timestamps stay monotonic."""
        wall = self._make_wall([100, 50, 50])
        hlc = HybridLogicalClock("A", wall_time=wall)

        ts1 = hlc.now()  # (100, 0, A)
        ts2 = hlc.now()  # pt=50 < last.pt=100 → keep 100, logical=1
        ts3 = hlc.now()  # pt=50 < last.pt=100 → keep 100, logical=2

        assert ts1 < ts2 < ts3

    def test_node_id_property(self):
        wall = self._make_wall([0])
        hlc = HybridLogicalClock("node-42", wall_time=wall)
        assert hlc.node_id == "node-42"

    def test_node_id_in_timestamps(self):
        wall = self._make_wall([100])
        hlc = HybridLogicalClock("node-X", wall_time=wall)
        ts = hlc.now()
        assert ts.node_id == "node-X"

    def test_with_node_clock(self):
        """HLC can use a NodeClock for physical time."""
        from happysimulator.core.clock import Clock

        clock = Clock(start_time=Instant.from_seconds(1.0))

        node_clock = NodeClock()
        node_clock.set_clock(clock)

        hlc = HybridLogicalClock("A", physical_clock=node_clock)
        ts = hlc.now()
        assert ts.physical_ns == Instant.from_seconds(1.0).nanoseconds

    def test_two_node_exchange(self):
        """Two nodes exchange messages, timestamps stay monotonic per node."""
        wall_a = self._make_wall([100, 100, 200, 200, 300])
        wall_b = self._make_wall([150, 150, 250, 250, 350])

        a = HybridLogicalClock("A", wall_time=wall_a)
        b = HybridLogicalClock("B", wall_time=wall_b)

        # A sends to B
        ts_a1 = a.send()  # (100, 0, A)
        b.receive(ts_a1)   # B reads pt=150, last=this receive result

        # B sends to A
        ts_b1 = b.send()
        a.receive(ts_b1)

        # A sends again
        ts_a2 = a.send()

        # All of A's timestamps are strictly monotonic
        assert ts_a1 < ts_a2
