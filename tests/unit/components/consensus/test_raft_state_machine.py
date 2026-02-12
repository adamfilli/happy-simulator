"""Tests for KVStateMachine."""

import pytest

from happysimulator.components.consensus import KVStateMachine, StateMachine


class TestSetAndGet:
    """Tests for set and get operations."""

    def test_set_and_get(self):
        """Set stores a value that can be retrieved with get."""
        sm = KVStateMachine()

        result = sm.apply({"op": "set", "key": "x", "value": 42})

        assert result == 42
        assert sm.apply({"op": "get", "key": "x"}) == 42

    def test_get_missing_key(self):
        """Get returns None for a key that does not exist."""
        sm = KVStateMachine()

        result = sm.apply({"op": "get", "key": "missing"})

        assert result is None


class TestDelete:
    """Tests for delete operation."""

    def test_delete(self):
        """Delete removes a key and returns its value."""
        sm = KVStateMachine()
        sm.apply({"op": "set", "key": "x", "value": 10})

        result = sm.apply({"op": "delete", "key": "x"})

        assert result == 10
        assert sm.apply({"op": "get", "key": "x"}) is None

    def test_delete_missing_key(self):
        """Delete on a missing key returns None."""
        sm = KVStateMachine()

        result = sm.apply({"op": "delete", "key": "nope"})

        assert result is None


class TestCompareAndSwap:
    """Tests for CAS (compare-and-swap) operation."""

    def test_cas_success(self):
        """CAS succeeds when expected matches current value."""
        sm = KVStateMachine()
        sm.apply({"op": "set", "key": "x", "value": "old"})

        result = sm.apply({"op": "cas", "key": "x", "expected": "old", "value": "new"})

        assert result is True
        assert sm.apply({"op": "get", "key": "x"}) == "new"

    def test_cas_failure(self):
        """CAS fails when expected does not match current value."""
        sm = KVStateMachine()
        sm.apply({"op": "set", "key": "x", "value": "actual"})

        result = sm.apply({"op": "cas", "key": "x", "expected": "wrong", "value": "new"})

        assert result is False
        assert sm.apply({"op": "get", "key": "x"}) == "actual"


class TestSnapshotAndRestore:
    """Tests for snapshot and restore."""

    def test_snapshot_and_restore(self):
        """Snapshot captures state and restore replaces it."""
        sm = KVStateMachine()
        sm.apply({"op": "set", "key": "a", "value": 1})
        sm.apply({"op": "set", "key": "b", "value": 2})

        snapshot = sm.snapshot()

        # Mutate state after snapshot
        sm.apply({"op": "set", "key": "c", "value": 3})
        sm.apply({"op": "delete", "key": "a"})

        # Restore to snapshot
        sm.restore(snapshot)

        assert sm.apply({"op": "get", "key": "a"}) == 1
        assert sm.apply({"op": "get", "key": "b"}) == 2
        assert sm.apply({"op": "get", "key": "c"}) is None


class TestInvalidCommands:
    """Tests for invalid command handling."""

    def test_invalid_command_no_op_key(self):
        """Command dict missing 'op' key raises ValueError."""
        sm = KVStateMachine()

        with pytest.raises(ValueError, match="Invalid command format"):
            sm.apply({"key": "x", "value": 1})

    def test_invalid_command_no_dict(self):
        """Non-dict command raises ValueError."""
        sm = KVStateMachine()

        with pytest.raises(ValueError, match="Invalid command format"):
            sm.apply("not a dict")

    def test_unknown_operation(self):
        """Unknown operation raises ValueError."""
        sm = KVStateMachine()

        with pytest.raises(ValueError, match="Unknown operation"):
            sm.apply({"op": "explode", "key": "x"})


class TestProtocol:
    """Tests for protocol compliance."""

    def test_implements_protocol(self):
        """KVStateMachine implements the StateMachine protocol."""
        sm = KVStateMachine()

        assert isinstance(sm, StateMachine)
