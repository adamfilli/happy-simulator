# Component Library Expansion Plan

## Context

The happysimulator component library currently has ~54 components across primitives (Queue, Resource, Sync), mid-level (rate limiters, resilience, cache, network), and high-level (Server, Client, LoadBalancer, ReplicatedStore, MessageQueue, etc.). The goal is to fill gaps across the full distributed systems stack -- from OS/hardware primitives up through consensus protocols, streaming systems, and operational patterns -- so users can simulate any distributed system scenario end-to-end.

---

## Phase 1: Storage Engine Internals
*Models the internals that determine real-world database performance.*

| Component | Type | Description |
|-----------|------|-------------|
| **WriteAheadLog** | Entity | Append-only durability log with sync policies (`SyncEveryWrite`, `SyncPeriodic`, `SyncOnBatch`). Uses `Resource` for disk I/O |
| **Memtable** | Entity | In-memory sorted write buffer, flushes to SSTable when full. Uses `RWLock` internally |
| **SSTable** | Pure class | Immutable sorted file with index + bloom filter. Read-only after creation |
| **LSMTree** | Entity | Full LSM engine composing Memtable + SSTable + WAL + BloomFilter. Pluggable compaction: `SizeTiered`, `Leveled`, `FIFO`. Exposes read/write/space amplification metrics |
| **BTree** | Entity | B-tree index with page reads/writes as I/O. Alternative to LSMTree for same interface |
| **TransactionManager** | Entity | Wraps any store with isolation levels: `ReadCommitted`, `SnapshotIsolation`, `Serializable`. Tracks read/write sets, detects conflicts, handles deadlocks |

**Simulation scenarios enabled:** Compaction storms causing latency spikes, read/write amplification tradeoffs, LSM vs B-tree under different workloads, transaction contention and deadlocks.

---

## Phase 2: Consensus & Coordination
*The foundation of reliable distributed systems.*

| Component | Type | Description |
|-----------|------|-------------|
| **PhiAccrualDetector** | Pure class | Probabilistic failure detector using heartbeat inter-arrival statistics. Configurable phi threshold |
| **MembershipProtocol** | Entity | SWIM-style gossip with probe, suspicion, and dissemination. Uses `Network` + `PhiAccrualDetector` |
| **LeaderElection** | Entity | Pluggable election strategies: `Bully`, `Ring`, `Randomized`. Uses `Network` + `MembershipProtocol` |
| **RaftNode** | Entity | Full Raft: leader election, log replication, snapshotting. Uses `Network` + `WAL` + `SimFuture` for election timeouts. Pluggable state machine |
| **PaxosNode** | Entity | Single-decree Paxos with proposer/acceptor/learner roles. Two-phase protocol (Prepare/Promise, Accept/Accepted). Uses `Network` + `SimFuture` for ballot contention |
| **MultiPaxosNode** | Entity | Multi-decree Paxos with stable leader optimization — skips Phase 1 when leader is established. Log-based with slot indexing. Uses `Network` + `WAL` |
| **FlexiblePaxosNode** | Entity | Paxos with asymmetric quorums (`Q1 + Q2 > N` instead of both being majorities). Configurable Phase 1 / Phase 2 quorum sizes for write-latency vs recovery-speed tradeoff |
| **DistributedLock** | Entity | Lock service with fencing tokens and lease-based expiry. Built on `RaftNode`, `MultiPaxosNode`, or standalone |

**Simulation scenarios enabled:** Split-brain under network partitions, election storms, consensus latency under load, fencing token correctness, failure detection tuning, Raft vs Multi-Paxos leader stability comparison, Flexible Paxos quorum tuning (write latency vs recovery time), ballot contention under competing proposers.

---

## Phase 3: Replication Protocols
*How data moves between replicas.*

| Component | Type | Description |
|-----------|------|-------------|
| **MerkleTree** | Pure class | Hash tree for anti-entropy sync. Compares trees to find divergent key ranges |
| **ConflictResolver** | Protocol + impls | `LastWriterWins(clock)`, `VectorClockMerge`, `CustomResolver(fn)` |
| **PrimaryBackupReplication** | Entity | Master-slave with `SyncReplication`, `SemiSync`, `AsyncReplication` modes. Exposes replication lag per backup |
| **ChainReplication** | Entity | Writes flow head-to-tail, reads from tail. CRAQ variant allows reads from any node |
| **MultiLeaderReplication** | Entity | Any node accepts writes. `VectorClock` for conflict detection, `ConflictResolver` for resolution, `MerkleTree` for anti-entropy |

**Simulation scenarios enabled:** Replication lag under load, data divergence during partitions, sync vs async durability tradeoffs, conflict resolution strategies compared.

---

## Phase 4: CRDTs (Distributed Data Structures)
*Eventually consistent data that converges automatically after partitions heal.*

| Component | Type | Description |
|-----------|------|-------------|
| **GCounter** | Pure class | Grow-only counter (per-node counters, merge via max) |
| **PNCounter** | Pure class | Positive-negative counter (two GCounters) |
| **LWWRegister** | Pure class | Last-Writer-Wins register using HLC timestamps |
| **ORSet** | Pure class | Observed-Remove Set with unique add tags |
| **CRDTStore** | Entity | KV store backed by CRDTs, gossip-based sync via `Network` + `MembershipProtocol`. Tracks convergence lag |

**Simulation scenarios enabled:** Eventual consistency during network partitions, convergence time after heal, CRDTs vs consensus comparison, shopping cart / counter scenarios.

---

## Phase 5: Streaming & Event Processing
*Kafka-like event streaming and stream processing.*

| Component | Type | Description |
|-----------|------|-------------|
| **EventLog** | Entity | Append-only partitioned log with retention policies. Uses `ShardingStrategy` for partition assignment. Tracks high watermarks per partition |
| **ConsumerGroup** | Entity | Coordinated consumers with partition assignment (`Range`, `RoundRobin`, `Sticky`) and rebalancing on join/leave |
| **StreamProcessor** | Entity | Stateful processing with windowing (`Tumbling`, `Sliding`, `Session`), watermarks, and late event handling |

**Simulation scenarios enabled:** Consumer rebalancing storms, backpressure propagation, exactly-once failures, window-based aggregation behavior, consumer lag analysis.

---

## Phase 6: Microservice Patterns
*Production patterns that compose existing components.*

| Component | Type | Description |
|-----------|------|-------------|
| **Sidecar** | Entity | Transparent proxy composing `CircuitBreaker` + `RateLimiter` + `Timeout` + `Retry` around any service |
| **APIGateway** | Entity | Request router with per-route rate limiting, backend pools, and auth simulation |
| **Saga** | Entity | Distributed transaction orchestrator: sequence of steps with compensating actions on failure |
| **OutboxRelay** | Entity | Transactional outbox pattern: write to DB in same tx, relay to `MessageQueue`/`EventLog` asynchronously |
| **IdempotencyStore** | Entity | Deduplication wrapper using idempotency keys with TTL |

**Simulation scenarios enabled:** Saga failure cascades, outbox relay lag, service mesh overhead, API gateway as bottleneck, idempotency under retries.

---

## Phase 7: Scheduling & Cluster Management
*Operational tooling and deployment patterns.*

| Component | Type | Description |
|-----------|------|-------------|
| **JobScheduler** | Entity | Cron-like periodic scheduling with priorities and DAG dependencies |
| **WorkStealingPool** | Entity | Per-worker deques with idle-worker stealing for tail latency reduction |
| **AutoScaler** | Entity | Metric-based scaling: `TargetUtilization`, `StepScaling`. Respects cooldown periods. Adds/removes backends from `LoadBalancer` |
| **RollingDeployer** | Entity | Gradual instance replacement with health checks and rollback |
| **CanaryDeployer** | Entity | Progressive traffic shifting (1% -> 5% -> 25% -> 100%) with automated rollback on degradation |

**Simulation scenarios enabled:** Scaling lag under sudden load, deployment-induced latency spikes, cold-start thundering herd, work-stealing vs round-robin tail latency.

---

## Phase 8: Low-Level Infrastructure Primitives
*OS and hardware-level models that affect distributed system behavior.*

| Component | Type | Description |
|-----------|------|-------------|
| **DiskIO** | Entity | Disk model with profiles: `HDD` (seek-sensitive), `SSD` (uniform), `NVMe` (high parallelism). Models queue depth effects |
| **PageCache** | Entity | OS page cache with LRU eviction, read-ahead, and writeback. Sits between storage engines and `DiskIO` |
| **CPUScheduler** | Entity | Time-slicing with policies: `FairShare`, `PriorityPreemptive`. Models context switch overhead |
| **GarbageCollector** | Entity | GC pause injection: `StopTheWorld`, `ConcurrentGC`, `GenerationalGC`. Injects pauses into any entity |
| **TCPConnection** | Entity | TCP-like transport with congestion control: `AIMD`, `Cubic`, `BBR`. Models slow start, cwnd, retransmission |
| **DNSResolver** | Entity | DNS with caching, TTL, hierarchical lookup latency |

**Simulation scenarios enabled:** GC pause cascading failures, disk I/O contention, TCP congestion collapse, CPU scheduling fairness, DNS cache miss storms.

---

## Build Order (by dependency and simulation value)

### Milestone A: Pure Algorithms (no dependencies)
PhiAccrualDetector, MerkleTree, GCounter, PNCounter, LWWRegister, ORSet, ConflictResolver

### Milestone B: Storage Engines (Phase 1)
WAL -> SSTable -> Memtable -> LSMTree -> BTree -> TransactionManager

### Milestone C: Consensus + Replication (Phases 2-3)
MembershipProtocol -> LeaderElection -> PaxosNode -> MultiPaxosNode -> FlexiblePaxosNode -> RaftNode -> DistributedLock -> PrimaryBackup -> ChainReplication -> MultiLeader

### Milestone D: CRDTs + Streaming (Phases 4-5)
CRDTStore -> EventLog -> ConsumerGroup -> StreamProcessor

### Milestone E: Microservice Patterns (Phase 6)
Sidecar, APIGateway, Saga, OutboxRelay, IdempotencyStore

### Milestone F: Operations + Infrastructure (Phases 7-8)
AutoScaler, RollingDeployer, CanaryDeployer, GarbageCollector, DiskIO, TCPConnection, etc.

---

## Total: 43 new components (8 pure algorithms + 35 entities)

## Verification
- Each component gets unit tests in `tests/unit/components/`
- Integration tests composing multiple components in `tests/integration/`
- **Every component gets a dedicated example in `examples/` that demonstrates its fundamental property** — e.g. LSMTree example shows write amplification during compaction, RaftNode example shows leader election and log convergence under partition, FlexiblePaxosNode example shows latency difference with asymmetric quorums. The example should make the "why does this component exist?" obvious to someone running it for the first time.
- Run `pytest -q` after each milestone to ensure no regressions
