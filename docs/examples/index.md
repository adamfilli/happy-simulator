# Examples

78 runnable examples across 10 categories. Each example is a self-contained Python script in the `examples/` directory.

| Category | Count | Description |
|----------|-------|-------------|
| [Queuing](queuing.md) | 7 | M/M/1 queues, metastable state, GC collapse, retrying clients |
| [Distributed Systems](distributed.md) | 12 | Raft, Paxos, CRDT, SWIM, chain replication, TCP congestion |
| [Industrial](industrial.md) | 20 | Bank, hospital, manufacturing, coffee shop, and more |
| [Infrastructure](infrastructure.md) | 7 | CPU scheduling, disk I/O, event logs, stream processing |
| [Storage](storage.md) | 7 | B-tree vs LSM, compaction, WAL, transactions |
| [Deployment](deployment.md) | 7 | Canary, rolling, saga, outbox, sidecar, GC cascade |
| [Performance](performance.md) | 7 | Auto scaler, API gateway, cold start, work stealing |
| [Behavior](behavior.md) | 2 | Product adoption, opinion dynamics |
| [Load Balancing](load-balancing.md) | 4 | Consistent hashing, vnodes, fleet changes, Zipf |
| [Visual Debugger](visual.md) | 1 | Browser-based debugger walkthrough |

## Running Examples

```bash
python examples/queuing/m_m_1_queue.py
python examples/visual/visual_debugger.py
```
