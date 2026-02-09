"""Unit tests for SimFuture, any_of, and all_of."""

import pytest

from happysimulator.core.sim_future import SimFuture, any_of, all_of


class TestSimFutureBasics:
    """Test SimFuture state management without a running simulation."""

    def test_initial_state(self):
        f = SimFuture()
        assert not f.is_resolved
        assert repr(f) == "SimFuture(pending)"

    def test_resolve_sets_value(self):
        f = SimFuture()
        f._resolved = True
        f._value = 42
        assert f.is_resolved
        assert f.value == 42

    def test_value_raises_if_not_resolved(self):
        f = SimFuture()
        with pytest.raises(RuntimeError, match="not been resolved"):
            _ = f.value

    def test_resolve_is_idempotent(self):
        """Resolving an already-resolved future is a no-op."""
        f = SimFuture()
        f._resolved = True
        f._value = 42
        f.resolve(99)  # Should be no-op
        assert f._value == 42

    def test_repr_states(self):
        f = SimFuture()
        assert "pending" in repr(f)

        f._resolved = True
        f._value = 42
        assert "resolved=42" in repr(f)


class TestSettleCallbacks:
    """Test the internal callback mechanism used by any_of/all_of."""

    def test_callback_fires_on_resolve(self):
        f = SimFuture()
        results = []
        f._add_settle_callback(lambda sf: results.append(sf._value))
        f._resolved = True
        f._value = "hello"
        f._fire_callbacks()
        assert results == ["hello"]

    def test_callback_fires_immediately_if_already_resolved(self):
        f = SimFuture()
        f._resolved = True
        f._value = "hello"
        results = []
        f._add_settle_callback(lambda sf: results.append(sf._value))
        assert results == ["hello"]

    def test_multiple_callbacks(self):
        f = SimFuture()
        results = []
        f._add_settle_callback(lambda sf: results.append("a"))
        f._add_settle_callback(lambda sf: results.append("b"))
        f._resolved = True
        f._value = 1
        f._fire_callbacks()
        assert results == ["a", "b"]

    def test_callbacks_cleared_after_firing(self):
        f = SimFuture()
        results = []
        f._add_settle_callback(lambda sf: results.append("x"))
        f._resolved = True
        f._fire_callbacks()
        f._fire_callbacks()  # Second call should not re-fire
        assert results == ["x"]


class TestAnyOf:
    """Test any_of combinator logic (without simulation)."""

    def test_requires_at_least_2(self):
        with pytest.raises(ValueError, match="at least 2"):
            any_of(SimFuture())

    def test_resolves_on_first(self):
        a, b = SimFuture(), SimFuture()
        composite = any_of(a, b)
        assert not composite.is_resolved

        a._resolved = True
        a._value = "first"
        a._fire_callbacks()

        assert composite.is_resolved
        assert composite._value == (0, "first")

    def test_second_resolve_is_ignored(self):
        a, b = SimFuture(), SimFuture()
        composite = any_of(a, b)

        a._resolved = True
        a._value = "first"
        a._fire_callbacks()

        b._resolved = True
        b._value = "second"
        b._fire_callbacks()

        assert composite._value == (0, "first")

    def test_already_resolved_future(self):
        a = SimFuture()
        a._resolved = True
        a._value = "already"
        b = SimFuture()

        composite = any_of(a, b)
        assert composite.is_resolved
        assert composite._value == (0, "already")


class TestAllOf:
    """Test all_of combinator logic (without simulation)."""

    def test_requires_at_least_2(self):
        with pytest.raises(ValueError, match="at least 2"):
            all_of(SimFuture())

    def test_resolves_when_all_resolve(self):
        a, b, c = SimFuture(), SimFuture(), SimFuture()
        composite = all_of(a, b, c)

        a._resolved = True
        a._value = 1
        a._fire_callbacks()
        assert not composite.is_resolved

        b._resolved = True
        b._value = 2
        b._fire_callbacks()
        assert not composite.is_resolved

        c._resolved = True
        c._value = 3
        c._fire_callbacks()
        assert composite.is_resolved
        assert composite._value == [1, 2, 3]

    def test_preserves_order(self):
        a, b = SimFuture(), SimFuture()
        composite = all_of(a, b)

        # Resolve in reverse order
        b._resolved = True
        b._value = "second"
        b._fire_callbacks()

        a._resolved = True
        a._value = "first"
        a._fire_callbacks()

        assert composite._value == ["first", "second"]

    def test_park_raises_if_already_parked(self):
        """Each SimFuture can only be yielded by one generator."""
        f = SimFuture()
        f._parked_process = object()
        f._parked_event_type = "test"

        class MockContinuation:
            process = object()
            event_type = "test2"
            daemon = False
            target = None
            on_complete = []
            context = {}

        with pytest.raises(RuntimeError, match="already has a parked process"):
            f._park(MockContinuation())
