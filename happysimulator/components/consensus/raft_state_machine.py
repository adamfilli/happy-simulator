"""State machine protocol and implementations for Raft/Paxos consensus.

Defines the interface for deterministic state machines that can be driven
by a replicated log. Commands from committed log entries are applied in
order to produce results.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StateMachine(Protocol):
    """Protocol for deterministic state machines driven by consensus.

    Commands from committed log entries are applied in order. The state
    machine must be deterministic: the same sequence of commands must
    always produce the same state.
    """

    def apply(self, command: Any) -> Any:
        """Apply a command and return the result.

        Args:
            command: The command to apply (implementation-defined format).

        Returns:
            The result of applying the command.
        """
        ...

    def snapshot(self) -> Any:
        """Capture the current state for snapshotting.

        Returns:
            A serializable representation of the current state.
        """
        ...

    def restore(self, snapshot: Any) -> None:
        """Restore state from a snapshot.

        Args:
            snapshot: A previously captured snapshot.
        """
        ...


class KVStateMachine:
    """Dictionary-backed key-value state machine.

    Commands are dicts with the following format:
        {"op": "set", "key": k, "value": v}  -> returns v
        {"op": "get", "key": k}               -> returns value or None
        {"op": "delete", "key": k}            -> returns deleted value or None
        {"op": "cas", "key": k, "expected": e, "value": v}  -> returns bool
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def apply(self, command: Any) -> Any:
        """Apply a key-value command.

        Args:
            command: Dict with 'op' key and operation-specific fields.

        Returns:
            Operation result.

        Raises:
            ValueError: If the command format is invalid.
        """
        if not isinstance(command, dict) or "op" not in command:
            raise ValueError(f"Invalid command format: {command}")

        op = command["op"]
        key = command.get("key")

        if op == "set":
            value = command.get("value")
            self._data[key] = value
            return value

        if op == "get":
            return self._data.get(key)

        if op == "delete":
            return self._data.pop(key, None)

        if op == "cas":
            expected = command.get("expected")
            current = self._data.get(key)
            if current == expected:
                self._data[key] = command.get("value")
                return True
            return False

        raise ValueError(f"Unknown operation: {op}")

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the current state."""
        return dict(self._data)

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        self._data = dict(snapshot)

    @property
    def data(self) -> dict[str, Any]:
        """Read-only view of the current state."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"KVStateMachine(keys={len(self._data)})"
