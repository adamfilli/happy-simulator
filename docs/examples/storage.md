# Storage Examples

7 examples exploring storage engine internals.

| Example | Description |
|---------|-------------|
| [btree_vs_lsm.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/storage/btree_vs_lsm.py) | B-tree vs LSM tree comparison under identical workloads |
| [lsm_compaction.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/storage/lsm_compaction.py) | Size-Tiered vs Leveled compaction strategy comparison |
| [memtable_flush.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/storage/memtable_flush.py) | Memtable write buffer filling and flushing to SSTable |
| [wal_sync_policies.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/storage/wal_sync_policies.py) | WAL sync policy comparison: throughput vs durability |
| [power_outage_durability.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/storage/power_outage_durability.py) | Power outage durability demonstration |
| [sstable_bloom_filter.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/storage/sstable_bloom_filter.py) | Bloom filter analysis: page reads saved for missing keys |
| [transaction_isolation.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/storage/transaction_isolation.py) | Transaction isolation level comparison under contention |

## Running

```bash
python examples/storage/btree_vs_lsm.py
```
