---
name: happy-sim-explain-example
description: Walk through a library example with detailed explanation
---

# Explain Example

Walk through a happysimulator example file, explaining how it works section by section.

## Available Examples

### queuing/
- `m_m_1_queue.py` — Classic M/M/1 queue with metastable failure demonstration
- `metastable_state.py` — True metastable failure with retrying client feedback loop
- `retrying_client.py` — Retry amplification causing sustained overload
- `increasing_queue_depth.py` — Stable vs unbounded queue growth under load ramp
- `dual_path_queue_latency.py` — Fast/slow path routing with queue-depth awareness
- `load_aware_routing.py` — Least-connections and weighted load distribution
- `gc_caused_collapse.py` — GC pauses triggering queue collapse

### distributed/
- `raft_leader_election.py` — Raft leader election, heartbeats, log replication
- `paxos_consensus.py` — Single-decree Paxos (Prepare/Promise, Accept/Accepted)
- `flexible_paxos_quorums.py` — Flexible Paxos with asymmetric quorums
- `crdt_convergence.py` — GCounter eventual consistency after partition
- `chain_replication.py` — Chain replication for strong consistency
- `primary_backup_replication.py` — Primary-backup with sync/async modes
- `multi_leader_replication.py` — Multi-leader with conflict resolution
- `distributed_lock_fencing.py` — Distributed lock with fencing tokens
- `swim_membership.py` — SWIM gossip-based failure detection
- `dns_cache_storm.py` — DNS cache expiration thundering herd
- `tcp_congestion.py` — TCP congestion control (Reno, Cubic, BBR)
- `degraded_network.py` — Impact of latency/loss on distributed performance

### industrial/
- `bank_branch.py` — Balking/reneging customers, shift-based staffing
- `manufacturing_line.py` — Assembly line with conveyor, inspection, rework loop
- `hospital_er.py` — ER triage with priority queuing
- `call_center.py` — IVR routing, agent skills, abandonment
- `coffee_shop.py` — Order queue, barista stations, drink prep
- `restaurant.py` — Host stand, tables, kitchen, meal courses
- `warehouse_fulfillment.py` — Order picking, packing, shipping zones
- `supply_chain.py` — Multi-tier with inventory policies
- `theme_park.py` — Attractions, FastPass, visitor routing
- `airport_terminal.py` — Security, check-in, boarding gates
- Plus: `car_wash`, `drive_through`, `grocery_store`, `hotel_operations`, `laundromat`, `parking_lot`, `pharmacy`, `blood_bank`, `elevator_system`, `urgent_care`

### infrastructure/
- `cpu_scheduling.py` — FairShare vs PriorityPreemptive scheduling
- `disk_io_contention.py` — Read/write queue priority
- `page_cache_eviction.py` — LRU/LFU eviction policies
- `consumer_group.py` — Kafka-style partition assignment
- `event_log.py` — Segment compaction, log-structured storage
- `stream_processor.py` — Windowing, aggregation, backpressure
- `job_scheduler_dag.py` — DAG dependency resolution

### storage/
- `btree_vs_lsm.py` — B-tree vs LSM read/write tradeoffs
- `lsm_compaction.py` — Size-tiered vs leveled compaction
- `wal_sync_policies.py` — WAL sync (every write, batch, async)
- `memtable_flush.py` — Flush policies and write latency spikes
- `sstable_bloom_filter.py` — Bloom filter effectiveness
- `transaction_isolation.py` — MVCC isolation levels
- `power_outage_durability.py` — WAL crash recovery

### deployment/
- `canary_deployment.py` — Progressive traffic shift with rollback
- `rolling_deployment.py` — Sequential server updates
- `saga_failure_cascade.py` — Distributed transaction compensation
- `service_mesh_sidecar.py` — Sidecar proxy with circuit breaking
- `gc_pause_cascade.py` — GC strategy impact on tail latency
- `idempotency_under_retries.py` — Idempotency tokens under retry storms
- `outbox_relay_lag.py` — Outbox pattern relay lag

### performance/
- `auto_scaler.py` — Scale-up/down policies with cooldown
- `api_gateway_bottleneck.py` — Per-route rate limiting
- `cold_start.py` — Serverless cold start with warm pool
- `inductor_burst_suppression.py` — EWMA burst smoothing
- `work_stealing_pool.py` — Work-stealing thread pool
- `zipf_cache_cohorts.py` — Cache behavior under Zipf access
- `metric_collection_pipeline.py` — Metrics buffering and batching
- `ai_analysis.py` — LLM-driven simulation analysis

### behavior/
- `product_adoption.py` — Innovator/majority adoption with social influence
- `opinion_dynamics.py` — DeGroot convergence and bounded-confidence clustering
- `adverse_advertising_amplification.py` — Negative advertising amplification

### load-balancing/
- `consistent_hashing_basics.py` — Consistent hashing vs round-robin cache affinity
- `fleet_change_comparison.py` — Hashing during server additions/removals
- `vnodes_analysis.py` — Virtual node count impact on load distribution
- `zipf_effect.py` — Zipf patterns with consistent hashing

### visual/
- `visual_debugger.py` — Bursty M/M/1 with browser visualization

## Instructions

1. If no example is specified, show the categories above and ask the user which one interests them.

2. Read the chosen example file completely.

3. Explain the example in these sections:

   **Overview** — What system is being simulated and why it's interesting (2-3 sentences).

   **Architecture** — What entities exist, how they connect, and how events flow through the pipeline. Mention which happysimulator components are used (e.g., `QueuedResource`, `Source.poisson()`, `Network`).

   **Key Patterns** — Highlight interesting library patterns used:
   - Generator yield forms (`yield delay`, `yield delay, [events]`, `yield future`)
   - SimFuture for request-response
   - Probes and Data for metrics
   - Network partitions or clock skew
   - Industrial components or behavioral agents

   **What to Watch For** — What the output/plots demonstrate. What insight does this example teach? (e.g., "Notice how the queue recovers from the spike in the stable case but collapses in the metastable case")

4. Optionally run the example if the user wants to see the output:
   ```
   python examples/<category>/<name>.py
   ```

5. Suggest related examples the user might want to explore next.
