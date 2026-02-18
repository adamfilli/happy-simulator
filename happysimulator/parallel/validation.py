"""Partition validation for parallel simulation.

Validates that partitions are truly independent: no entity appears in
multiple partitions, no entity references entities in other partitions,
and no source targets a cross-partition entity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.core.protocols import Simulatable

if TYPE_CHECKING:
    from happysimulator.parallel.partition import SimulationPartition

logger = logging.getLogger(__name__)


def validate_partitions(partitions: list[SimulationPartition]) -> None:
    """Validate that partitions are independent.

    Checks:
        1. No entity appears in multiple partitions.
        2. No source targets an entity outside its partition.
        3. No entity attribute references an entity in another partition.

    Args:
        partitions: The partitions to validate.

    Raises:
        ValueError: If any cross-partition dependency is detected.
    """
    if not partitions:
        raise ValueError("At least one partition is required")

    names = [p.name for p in partitions]
    if len(names) != len(set(names)):
        dupes = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate partition names: {set(dupes)}")

    # Build membership map: id(entity) -> (partition_name, entity_name)
    membership: dict[int, tuple[str, str]] = {}

    for part in partitions:
        for entity in part.entities:
            eid = id(entity)
            ename = getattr(entity, "name", repr(entity))
            if eid in membership:
                prev_part, prev_name = membership[eid]
                raise ValueError(
                    f"Entity '{ename}' appears in both partition "
                    f"'{prev_part}' and '{part.name}'"
                )
            membership[eid] = (part.name, ename)

    # Validate source targets
    for part in partitions:
        for source in part.sources:
            _check_source_target(source, part.name, membership)

    # Walk entity attributes for cross-references
    for part in partitions:
        for entity in part.entities:
            _check_entity_references(entity, part.name, membership)


def _check_source_target(
    source: object,
    partition_name: str,
    membership: dict[int, tuple[str, str]],
) -> None:
    """Check that a source's target entity is in the same partition."""
    # Sources store their target in various attributes
    for attr in ("_target", "target"):
        target = getattr(source, attr, None)
        if target is not None and isinstance(target, (Entity, Simulatable)):
            tid = id(target)
            if tid in membership:
                target_part, target_name = membership[tid]
                if target_part != partition_name:
                    source_name = getattr(source, "name", repr(source))
                    raise ValueError(
                        f"Source '{source_name}' in partition '{partition_name}' "
                        f"targets entity '{target_name}' in partition "
                        f"'{target_part}'"
                    )
            break


def _check_entity_references(
    entity: object,
    partition_name: str,
    membership: dict[int, tuple[str, str]],
    _visited: set[int] | None = None,
    _depth: int = 0,
    _max_depth: int = 3,
) -> None:
    """Walk entity attributes for references to entities in other partitions.

    Recurses into sub-entities (e.g., QueuedResource internals) up to
    _max_depth to catch composition patterns.
    """
    if _depth >= _max_depth:
        return
    if _visited is None:
        _visited = set()
    eid = id(entity)
    if eid in _visited:
        return
    _visited.add(eid)

    entity_name = getattr(entity, "name", repr(entity))

    for attr_name in _get_instance_attrs(entity):
        if attr_name.startswith("_clock"):
            continue  # Clock is expected to be shared within a partition

        value = getattr(entity, attr_name, None)

        if isinstance(value, (Entity, Simulatable)) and hasattr(value, "name"):
            _check_single_ref(
                value, entity_name, attr_name, partition_name, membership,
            )
            # Recurse into sub-entities within the same partition
            vid = id(value)
            if vid in membership and membership[vid][0] == partition_name:
                _check_entity_references(
                    value, partition_name, membership,
                    _visited, _depth + 1, _max_depth,
                )

        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (Entity, Simulatable)) and hasattr(item, "name"):
                    _check_single_ref(
                        item, entity_name, attr_name, partition_name, membership,
                    )

        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, (Entity, Simulatable)) and hasattr(item, "name"):
                    _check_single_ref(
                        item, entity_name, attr_name, partition_name, membership,
                    )


def _check_single_ref(
    ref_entity: object,
    owner_name: str,
    attr_name: str,
    partition_name: str,
    membership: dict[int, tuple[str, str]],
) -> None:
    """Raise if ref_entity belongs to a different partition."""
    rid = id(ref_entity)
    if rid in membership:
        ref_part, ref_name = membership[rid]
        if ref_part != partition_name:
            raise ValueError(
                f"Entity '{owner_name}' in partition '{partition_name}' "
                f"references entity '{ref_name}' in partition "
                f"'{ref_part}' via attribute '{attr_name}'"
            )


def _get_instance_attrs(obj: object) -> list[str]:
    """Get instance attribute names, handling __slots__ and __dict__."""
    attrs: list[str] = []
    if hasattr(obj, "__dict__"):
        attrs.extend(obj.__dict__.keys())
    # Walk __slots__ up the MRO
    for cls in type(obj).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot not in attrs and not slot.startswith("__"):
                attrs.append(slot)
    return attrs


def build_entity_id_set(partition: SimulationPartition) -> frozenset[int]:
    """Build a set of entity ids for runtime cross-partition detection.

    Includes top-level entities, sources, probes, and internal sub-entities
    discovered via attribute walking (e.g., QueuedResource internals).
    Sources and probes are included because they receive their own internal
    events (e.g., SourceEvent for scheduling the next arrival).
    """
    ids: set[int] = set()
    for entity in partition.entities:
        _collect_entity_ids(entity, ids)
    for source in partition.sources:
        _collect_entity_ids(source, ids)
    for probe in partition.probes:
        _collect_entity_ids(probe, ids)
    return frozenset(ids)


def _collect_entity_ids(
    entity: object,
    ids: set[int],
    _depth: int = 0,
    _max_depth: int = 4,
) -> None:
    """Recursively collect entity ids from an entity and its sub-entities."""
    eid = id(entity)
    if eid in ids or _depth >= _max_depth:
        return
    ids.add(eid)

    for attr_name in _get_instance_attrs(entity):
        if attr_name.startswith("_clock"):
            continue
        value = getattr(entity, attr_name, None)
        if isinstance(value, (Entity, Simulatable)) and hasattr(value, "handle_event"):
            _collect_entity_ids(value, ids, _depth + 1, _max_depth)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (Entity, Simulatable)) and hasattr(item, "handle_event"):
                    _collect_entity_ids(item, ids, _depth + 1, _max_depth)
