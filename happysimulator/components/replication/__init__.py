"""Replication protocol components for distributed systems simulation.

Provides components that model how data moves between replicas:

- **ConflictResolver**: Protocol + implementations (LWW, VectorClockMerge, Custom)
- **PrimaryBackupReplication**: Master-slave with sync/semi-sync/async modes
- **ChainReplication**: Head-to-tail writes, tail reads, CRAQ variant
- **MultiLeaderReplication**: Any-node writes with vector clock conflicts
"""

from happysimulator.components.replication.conflict_resolver import (
    ConflictResolver,
    CustomResolver,
    LastWriterWins,
    VectorClockMerge,
    VersionedValue,
)
from happysimulator.components.replication.primary_backup import (
    BackupNode,
    BackupStats,
    PrimaryBackupStats,
    PrimaryNode,
    ReplicationMode,
)
from happysimulator.components.replication.chain_replication import (
    ChainNode,
    ChainNodeRole,
    ChainReplicationStats,
    build_chain,
)
from happysimulator.components.replication.multi_leader import (
    LeaderNode,
    MultiLeaderStats,
)

__all__ = [
    # Conflict resolution
    "ConflictResolver",
    "CustomResolver",
    "LastWriterWins",
    "VectorClockMerge",
    "VersionedValue",
    # Primary-backup
    "PrimaryNode",
    "BackupNode",
    "ReplicationMode",
    "PrimaryBackupStats",
    "BackupStats",
    # Chain replication
    "ChainNode",
    "ChainNodeRole",
    "ChainReplicationStats",
    "build_chain",
    # Multi-leader
    "LeaderNode",
    "MultiLeaderStats",
]
