"""Storage engine internals for database simulation.

Provides components modeling the internal data structures of storage
engines: WAL, Memtable, SSTable, LSM Tree, B-Tree, and Transaction
Manager. These enable simulation of compaction storms, read/write
amplification tradeoffs, and transaction contention.
"""

from happysimulator.components.storage.sstable import SSTable, SSTableStats
from happysimulator.components.storage.wal import (
    WriteAheadLog,
    WALEntry,
    WALStats,
    SyncPolicy,
    SyncEveryWrite,
    SyncPeriodic,
    SyncOnBatch,
)
from happysimulator.components.storage.memtable import Memtable, MemtableStats
from happysimulator.components.storage.lsm_tree import (
    LSMTree,
    LSMTreeStats,
    CompactionStrategy,
    SizeTieredCompaction,
    LeveledCompaction,
    FIFOCompaction,
)
from happysimulator.components.storage.btree import BTree, BTreeStats
from happysimulator.components.storage.transaction_manager import (
    TransactionManager,
    StorageTransaction,
    TransactionStats,
    IsolationLevel,
    StorageEngine,
)

__all__ = [
    # SSTable
    "SSTable",
    "SSTableStats",
    # WAL
    "WriteAheadLog",
    "WALEntry",
    "WALStats",
    "SyncPolicy",
    "SyncEveryWrite",
    "SyncPeriodic",
    "SyncOnBatch",
    # Memtable
    "Memtable",
    "MemtableStats",
    # LSM Tree
    "LSMTree",
    "LSMTreeStats",
    "CompactionStrategy",
    "SizeTieredCompaction",
    "LeveledCompaction",
    "FIFOCompaction",
    # B-Tree
    "BTree",
    "BTreeStats",
    # Transaction Manager
    "TransactionManager",
    "StorageTransaction",
    "TransactionStats",
    "IsolationLevel",
    "StorageEngine",
]
