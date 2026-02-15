"""Conflict resolution strategies for multi-leader replication.

Provides a protocol and implementations for resolving write conflicts when
the same key is written concurrently on multiple replicas. Three strategies:

- **LastWriterWins**: Highest timestamp wins (Cassandra, DynamoDB).
- **VectorClockMerge**: Causal ordering wins; concurrent values delegate
  to a merge function (Riak, Voldemort).
- **CustomResolver**: User-supplied function for application-specific logic.

Example::

    from happysimulator.components.replication import (
        LastWriterWins,
        VectorClockMerge,
        VersionedValue,
    )

    resolver = LastWriterWins()
    winner = resolver.resolve(
        "user:123",
        [
            VersionedValue(value={"name": "Alice"}, timestamp=1.0, writer_id="dc-east"),
            VersionedValue(value={"name": "Bob"}, timestamp=2.0, writer_id="dc-west"),
        ],
    )
    assert winner.value == {"name": "Bob"}  # Higher timestamp wins
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from happysimulator.core.logical_clocks import HLCTimestamp

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class VersionedValue:
    """A value with version metadata for conflict resolution.

    Attributes:
        value: The stored value.
        timestamp: Write timestamp (float seconds or HLCTimestamp).
        writer_id: Identifier of the node that wrote this version.
        vector_clock: Optional vector clock snapshot at write time.
    """

    value: Any
    timestamp: float | HLCTimestamp
    writer_id: str
    vector_clock: dict[str, int] | None = None


@runtime_checkable
class ConflictResolver(Protocol):
    """Protocol for resolving conflicting versions of a key.

    Implementations choose a winning version when multiple replicas
    have divergent values for the same key.
    """

    def resolve(self, key: str, versions: list[VersionedValue]) -> VersionedValue:
        """Choose a winning version from conflicting versions.

        Args:
            key: The key being resolved.
            versions: Two or more conflicting versions (never empty).

        Returns:
            The chosen winning version.
        """
        ...


class LastWriterWins:
    """Conflict resolver that picks the version with the highest timestamp.

    Supports both float timestamps and HLCTimestamp. For HLCTimestamp,
    uses the built-in comparison (physical_ns, logical, node_id).
    For float timestamps, the highest value wins. Ties are broken by
    writer_id for determinism.

    This is the simplest strategy, used by Cassandra and DynamoDB.
    Data loss is possible when concurrent writes have close timestamps.
    """

    def resolve(self, key: str, versions: list[VersionedValue]) -> VersionedValue:
        """Pick the version with the highest timestamp.

        Args:
            key: The key being resolved.
            versions: Conflicting versions.

        Returns:
            The version with the highest timestamp.
        """
        return max(versions, key=self._sort_key)

    @staticmethod
    def _sort_key(v: VersionedValue) -> tuple:
        ts = v.timestamp
        if isinstance(ts, HLCTimestamp):
            return (ts.physical_ns, ts.logical, ts.node_id)
        return (ts, 0, v.writer_id)


class VectorClockMerge:
    """Conflict resolver using vector clock causal ordering.

    If one version's vector clock causally dominates, that version wins.
    If the versions are concurrent (neither dominates), delegates to a
    user-supplied ``merge_fn``. If no merge function is provided,
    falls back to the version with the highest timestamp.

    Args:
        merge_fn: Optional function called for concurrent versions.
            Signature: (key, version_a, version_b) -> VersionedValue.
    """

    def __init__(
        self,
        merge_fn: Callable[[str, VersionedValue, VersionedValue], VersionedValue] | None = None,
    ):
        self._merge_fn = merge_fn

    def resolve(self, key: str, versions: list[VersionedValue]) -> VersionedValue:
        """Resolve using vector clock ordering, merging concurrent versions.

        Args:
            key: The key being resolved.
            versions: Conflicting versions (must have vector_clock set).

        Returns:
            The causally dominant version, or the merge result for concurrent versions.
        """
        result = versions[0]
        for v in versions[1:]:
            result = self._resolve_pair(key, result, v)
        return result

    def _resolve_pair(self, key: str, a: VersionedValue, b: VersionedValue) -> VersionedValue:
        """Resolve a pair of versions."""
        vc_a = a.vector_clock or {}
        vc_b = b.vector_clock or {}

        if _vc_dominates(vc_a, vc_b):
            return a
        if _vc_dominates(vc_b, vc_a):
            return b

        # Concurrent — use merge function or fall back to LWW
        if self._merge_fn is not None:
            return self._merge_fn(key, a, b)

        # Fallback: highest timestamp wins
        return LastWriterWins().resolve(key, [a, b])


class CustomResolver:
    """Conflict resolver wrapping a user-supplied function.

    Args:
        resolve_fn: Function with signature (key, versions) -> VersionedValue.
    """

    def __init__(
        self,
        resolve_fn: Callable[[str, list[VersionedValue]], VersionedValue],
    ):
        self._resolve_fn = resolve_fn

    def resolve(self, key: str, versions: list[VersionedValue]) -> VersionedValue:
        """Delegate to the user-supplied function.

        Args:
            key: The key being resolved.
            versions: Conflicting versions.

        Returns:
            The version chosen by the user function.
        """
        return self._resolve_fn(key, versions)


def _vc_dominates(a: dict[str, int], b: dict[str, int]) -> bool:
    """Check if vector clock ``a`` causally dominates ``b``.

    Returns True if all components of ``a`` >= corresponding components
    of ``b``, and at least one is strictly greater.
    """
    all_keys = set(a) | set(b)
    all_geq = True
    any_gt = False
    for k in all_keys:
        val_a = a.get(k, 0)
        val_b = b.get(k, 0)
        if val_a < val_b:
            all_geq = False
            break
        if val_a > val_b:
            any_gt = True
    return all_geq and any_gt
