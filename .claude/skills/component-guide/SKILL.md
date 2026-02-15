---
name: component-guide
description: Help choose the right happysimulator components for a use case
---

# Component Guide

Interactive wizard to help users pick the right happysimulator components for their simulation.

## Instructions

1. Ask the user what system they want to model if not specified. Get a brief description of the scenario (e.g., "a web service with retries", "a factory with breakdowns", "a social network with opinion spread").

2. Read the project's CLAUDE.md for the full component catalog.

3. Map the user's scenario to the best-fit components. Use this decision tree:

### Does it have a queue + processing?
→ **`QueuedResource`** (override `handle_queued_event()` and `has_capacity()`)
→ With priority? Use `PriorityQueue` policy
→ With balking? Use `BalkingQueue` policy
→ With reneging? Use `RenegingQueuedResource`

### Does it need load generation?
→ Deterministic: **`Source.constant(rate=N, target=entity)`**
→ Stochastic: **`Source.poisson(rate=N, target=entity)`**
→ Time-varying: **`Source.with_profile(profile=MyProfile(), target=entity)`**
→ Custom events: Implement `EventProvider` and pass to `Source()`

### Does it involve a network of nodes?
→ **`Network`** + `add_bidirectional_link()` + link condition factories
→ Need partitions? `network.partition([a], [b])` / `partition.heal()`
→ Need clock skew? **`NodeClock(FixedSkew(...))`** or **`NodeClock(LinearDrift(...))`**
→ Need causal ordering? **`LamportClock`**, **`VectorClock`**, or **`HybridLogicalClock`**

### Does it need consensus or replication?
→ Leader election + log replication: **`RaftNode`**
→ Classic consensus: **`PaxosNode`** or **`FlexiblePaxosNode`**
→ Eventual consistency: **`CRDTStore`** with `GCounter`, `PNCounter`, `LWWRegister`, `ORSet`
→ Primary-backup: **`PrimaryNode`** + **`BackupNode`**
→ Chain replication: **`ChainNode`**

### Does it need resilience patterns?
→ Fail-fast after errors: **`CircuitBreaker`**
→ Limit concurrency: **`Bulkhead`**
→ Limit wait time: **`TimeoutWrapper`**
→ Backup request: **`Hedge`**
→ Graceful degradation: **`Fallback`**

### Does it need rate limiting?
→ Token bucket: **`RateLimitedEntity`** + **`TokenBucketPolicy`**
→ Leaky bucket: **`RateLimitedEntity`** + **`LeakyBucketPolicy`**
→ Burst suppression (no throughput cap): **`Inductor`**
→ Adaptive: **`RateLimitedEntity`** + **`AdaptivePolicy`**

### Does it need contended resources?
→ **`Resource("name", capacity=N)`** + `yield resource.acquire(amount)` + `grant.release()`
→ With preemption? **`PreemptibleResource`**
→ Pooled with fixed cycle? **`PooledCycleResource`**

### Is it an industrial/operations scenario?
→ Assembly line: **`ConveyorBelt`** + **`InspectionStation`**
→ Batch processing: **`BatchProcessor`**
→ Shift-based staffing: **`ShiftSchedule`** + **`ShiftedServer`**
→ Equipment breakdowns: **`BreakdownScheduler`**
→ Inventory management: **`InventoryBuffer`** or **`PerishableInventory`**
→ Scheduled arrivals: **`AppointmentScheduler`**
→ Gate/valve control: **`GateController`**
→ Fan-out/fan-in: **`SplitMerge`**
→ Conditional routing: **`ConditionalRouter`**

### Is it behavioral / agent-based?
→ Individual agents: **`Agent`** + `PersonalityTraits` + decision model
→ Population: **`Population.uniform()`** or **`Population.from_segments()`**
→ Social influence: **`Environment`** + influence model (`DeGrootModel`, `BoundedConfidenceModel`, `VoterModel`)
→ Stimuli: `broadcast_stimulus()`, `price_change()`, `influence_propagation()`

### Does it need storage / database modeling?
→ Key-value: **`KVStore`**
→ With cache: **`CachedStore`**
→ Sharded: **`ShardedStore`**
→ LSM tree: **`LSMTree`** + compaction strategies
→ Transactions: **`TransactionManager`** with `IsolationLevel`

### Does it need messaging / streaming?
→ Pub/sub: **`MessageQueue`** + **`Topic`**
→ Event log: **`EventLog`** + **`ConsumerGroup`**
→ Stream processing: **`StreamProcessor`**
→ Dead letters: **`DeadLetterQueue`**

### What should collect results?
→ Latency tracking: **`Sink`** (auto-tracks from `context["created_at"]`) or **`LatencyTracker`**
→ Event counting: **`Counter`**
→ Throughput: **`ThroughputTracker`**
→ Time series: **`Probe`** + **`Data`**

4. Present the recommended components with:
   - A brief explanation of why each was chosen
   - A minimal wiring example showing how they connect
   - The most relevant example file(s) from `examples/` for reference

5. Ask if the user wants to scaffold the full simulation (`/scaffold`).
