"""Init-time validation for parallel simulation partitions.

Catches misconfigurations early — entity duplication, source target
misplacement, illegal cross-partition references, and link consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from happysimulator.core.callback_entity import CallbackEntity
from happysimulator.core.protocols import Simulatable

if TYPE_CHECKING:
    from happysimulator.parallel.link import PartitionLink
    from happysimulator.parallel.partition import SimulationPartition


def validate_partitions(
    partitions: list[SimulationPartition],
    links: list[PartitionLink] | None = None,
    window_size: float | None = None,
) -> None:
    """Validate partition declarations and link consistency.

    Raises:
        ValueError: On any detected misconfiguration.
    """
    links = links or []
    partition_names = {p.name for p in partitions}

    # 1. Unique partition names
    if len(partition_names) != len(partitions):
        seen: set[str] = set()
        for p in partitions:
            if p.name in seen:
                raise ValueError(f"Duplicate partition name: '{p.name}'")
            seen.add(p.name)

    # 2. No entity in multiple partitions
    entity_to_partition: dict[int, str] = {}
    for p in partitions:
        for entity in p.entities:
            eid = id(entity)
            if eid in entity_to_partition:
                entity_name = getattr(entity, "name", repr(entity))
                raise ValueError(
                    f"Entity '{entity_name}' is in partitions "
                    f"'{entity_to_partition[eid]}' and '{p.name}'"
                )
            entity_to_partition[eid] = p.name

    # 3. Source targets must be in same partition
    for p in partitions:
        partition_eids = {id(e) for e in p.entities}
        for source in p.sources:
            target = getattr(source, "_target", None)
            if target is None:
                # Try to get from event_provider
                ep = getattr(source, "_event_provider", None)
                if ep is not None:
                    target = getattr(ep, "_target", None)
            if target is not None and id(target) not in partition_eids:
                # Check if target is in any partition
                target_partition = entity_to_partition.get(id(target))
                if target_partition is not None and target_partition != p.name:
                    source_name = getattr(source, "name", repr(source))
                    target_name = getattr(target, "name", repr(target))
                    raise ValueError(
                        f"Source '{source_name}' in partition '{p.name}' "
                        f"targets entity '{target_name}' in partition "
                        f"'{target_partition}'"
                    )

    # 4. Link partition names must exist
    for link in links:
        if link.source_partition not in partition_names:
            raise ValueError(
                f"PartitionLink references unknown source partition "
                f"'{link.source_partition}'"
            )
        if link.dest_partition not in partition_names:
            raise ValueError(
                f"PartitionLink references unknown dest partition "
                f"'{link.dest_partition}'"
            )

    # 5. Build set of linked partition pairs (directed)
    linked_pairs: set[tuple[str, str]] = set()
    for link in links:
        linked_pairs.add((link.source_partition, link.dest_partition))

    # 6. Cross-partition entity references
    for p in partitions:
        partition_eids = {id(e) for e in p.entities}
        for entity in p.entities:
            _check_cross_references(
                entity, p.name, partition_eids,
                entity_to_partition, linked_pairs, depth=0,
            )

    # 7. Window size constraint
    if window_size is not None and links:
        min_link_latency = min(link.min_latency for link in links)
        if window_size > min_link_latency:
            raise ValueError(
                f"window_size ({window_size}s) must be <= "
                f"min(link.min_latency) ({min_link_latency}s)"
            )


_MAX_WALK_DEPTH = 3


def _check_cross_references(
    obj: object,
    partition_name: str,
    partition_eids: set[int],
    entity_to_partition: dict[int, str],
    linked_pairs: set[tuple[str, str]],
    depth: int,
) -> None:
    """Walk entity attributes (up to _MAX_WALK_DEPTH) for cross-partition refs."""
    if depth >= _MAX_WALK_DEPTH:
        return

    # Skip CallbackEntity — always local
    if isinstance(obj, CallbackEntity):
        return

    attrs = vars(obj) if hasattr(obj, "__dict__") else {}
    for attr_name, attr_val in attrs.items():
        if attr_name.startswith("_"):
            continue
        _check_value(
            attr_val, partition_name, partition_eids,
            entity_to_partition, linked_pairs, depth,
        )


def _check_value(
    val: object,
    partition_name: str,
    partition_eids: set[int],
    entity_to_partition: dict[int, str],
    linked_pairs: set[tuple[str, str]],
    depth: int,
) -> None:
    """Check a single value for cross-partition entity references."""
    if isinstance(val, Simulatable):
        _check_entity_ref(
            val, partition_name, partition_eids,
            entity_to_partition, linked_pairs,
        )
    elif isinstance(val, (list, tuple)):
        for item in val:
            if isinstance(item, Simulatable):
                _check_entity_ref(
                    item, partition_name, partition_eids,
                    entity_to_partition, linked_pairs,
                )
    elif isinstance(val, dict):
        for item in val.values():
            if isinstance(item, Simulatable):
                _check_entity_ref(
                    item, partition_name, partition_eids,
                    entity_to_partition, linked_pairs,
                )
    elif hasattr(val, "__dict__") and not isinstance(val, type):
        _check_cross_references(
            val, partition_name, partition_eids,
            entity_to_partition, linked_pairs, depth + 1,
        )


def _check_entity_ref(
    entity: Simulatable,
    partition_name: str,
    partition_eids: set[int],
    entity_to_partition: dict[int, str],
    linked_pairs: set[tuple[str, str]],
) -> None:
    """Raise if entity is in another partition without a link."""
    eid = id(entity)
    if eid in partition_eids:
        return  # same partition — OK

    target_partition = entity_to_partition.get(eid)
    if target_partition is None:
        return  # entity not in any partition (e.g. not registered)

    # Cross-partition ref — check if linked
    if (partition_name, target_partition) not in linked_pairs:
        entity_name = getattr(entity, "name", repr(entity))
        raise ValueError(
            f"Entity in partition '{partition_name}' references "
            f"entity '{entity_name}' in partition '{target_partition}' "
            f"but no PartitionLink exists from '{partition_name}' to "
            f"'{target_partition}'"
        )


def build_entity_sets(
    partitions: list[SimulationPartition],
) -> dict[str, frozenset[int]]:
    """Build partition membership maps keyed by partition name.

    Includes entities, sources, and probes since sources self-schedule
    events targeting themselves.

    Returns:
        Mapping from partition name to frozenset of entity ``id()`` values.
    """
    result: dict[str, frozenset[int]] = {}
    for p in partitions:
        ids: set[int] = set()
        for e in p.entities:
            ids.add(id(e))
        for s in p.sources:
            ids.add(id(s))
        for pr in p.probes:
            ids.add(id(pr))
        result[p.name] = frozenset(ids)
    return result
