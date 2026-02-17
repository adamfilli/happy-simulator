# Simulation-as-Specification for Verified Distributed Systems

## Context

You're asking whether simulation can serve as the **unifying abstraction** that bridges formal specification and real implementation for distributed systems — covering not just logical correctness (like TLA+) but also latency, availability, hardware behavior, and deployment/change management.

**Short answer: Yes, you're on the right track. Simulation is the most promising medium for this.** But the hard problem isn't the simulation — it's the *bridge* between simulation and implementation. Below is a detailed analysis of why, what exists, what's missing, and what the architecture should look like.

---

## The Landscape: What Exists and Where Each Falls Short

### TLA+ / Alloy
- **What it does**: Exhaustively model-checks all possible behaviors of a protocol
- **Gap**: Pure logic — no timing, no latency distributions, no deployment topology. And critically: **no connection to implementation whatsoever**. You prove the spec, then manually write Go/Rust and *hope* it matches.

### P Language (Microsoft)
- **What it does**: State-machine-based spec language that *generates executable C#/C code*. Used in AWS S3's consistency proof, Windows USB 3.0 drivers.
- **Gap**: Closest to your vision, but limited to state machine protocols. No timing model, no deployment model, no latency distributions. The generated code is a protocol skeleton — real systems need much more around it. No runtime conformance checking.

### FoundationDB (Deterministic Simulation Testing)
- **What it does**: Real C++ code runs inside a deterministic simulation. Same binary in prod and test. Fault injection (BUGGIFY) randomizes failures. ~1 trillion CPU-hours of testing.
- **Gap**: **Testing, not proving.** Finds bugs with high probability but can't prove absence. No formal spec — the code IS the spec (which means if the code is wrong, there's nothing to catch it). No deployment model.

### IronFleet / Verdi (Verified Distributed Systems)
- **What it does**: Proves implementation refines spec using Dafny/Coq. IronFleet proved both safety AND liveness for a Paxos implementation — first ever.
- **Gap**: Enormous proof effort (person-years). No timing model — proofs are about logical correctness only. Can't express "p99 latency < 10ms" or "survives rolling deployment."

### Digital Twins + Runtime Verification
- **What it does**: Simulation runs alongside production; runtime monitors check invariants expressed in temporal logic.
- **Gap**: Focused on cyber-physical systems (factories, satellites), not distributed software. No formal refinement proofs. Drift detection is reactive, not preventive.

### Summary Table

| Approach | Logical Correctness | Timing/Perf | Deployment | Impl Connection | Proof Strength |
|----------|:------------------:|:-----------:|:----------:|:---------------:|:--------------:|
| TLA+ | **Full** | None | None | None | Exhaustive proof |
| P | Partial | None | None | Code gen | Model checking |
| FoundationDB | Via testing | Via testing | None | Same binary | Probabilistic |
| IronFleet | **Full** | None | None | Refinement proof | Mathematical proof |
| Digital Twin | Via monitoring | Real-time | None | Runtime monitoring | Runtime only |
| **Your vision** | **Full** | **Full** | **Full** | **Bidirectional** | **Layered** |

---

## Why Simulation Is the Right Unifying Abstraction

1. **Timing is first-class**: Latency distributions, network jitter, clock skew — all natural in a DES. TLA+ can't express "this message takes 5-50ms"; a simulation models it trivially.

2. **Deployment is modelable**: A rolling update IS a sequence of events (stop node, drain connections, deploy new binary, health check, route traffic). This is exactly what a simulation models.

3. **Hardware is modelable**: Disk IOPS, memory pressure, CPU contention, NIC queuing — all expressible as entity behaviors with timing.

4. **It's executable**: Unlike TLA+, you can run it, watch it, debug it. Unlike prose specs, it's unambiguous.

5. **It's the natural oracle**: Given the same inputs and failures, the simulation tells you what *should* happen. Compare that to what *does* happen.

---

## The Assurance Question: How Strong Can the Guarantees Actually Be?

The naive answer is "Rice's theorem — you can't prove implementation matches spec for Turing-complete languages." But that's overly pessimistic. With the right language design, you can get **much stronger guarantees than testing alone** for most properties. The key is matching **each property type to its strongest feasible verification method**:

| Property | Example | Strongest Feasible Method | Strength |
|----------|---------|--------------------------|----------|
| Safety invariants | "At most one leader per term" | Model checking (extract to TLA+) | **Exhaustive proof** |
| Liveness | "A leader is eventually elected" | Temporal logic proof (IronFleet showed this works) | **Mathematical proof** |
| Protocol conformance | "Messages follow request→response→ack pattern" | **Session types** (compile-time!) | **Type-system guarantee** |
| Deployment safety | "Rolling update keeps ≥2/3 replicas available" | Model checking (deployment IS finite-state) | **Exhaustive proof** |
| Timing/performance | "p99 latency < 10ms under 1000 req/s" | Statistical simulation (hardware-dependent) | **Statistical bounds** |
| Custom business logic | "Order total calculated correctly" | Refinement proofs OR runtime monitoring | Varies |
| Production drift | "v2.3 behaves same as spec" | Runtime trace conformance | **Continuous monitoring** |

### The critical insight: ~80% of what you care about IS formally provable

- **Safety, liveness, protocol conformance, deployment safety** — all amenable to formal proof if the spec language is designed right
- **Timing/performance** — genuinely cannot be proven (depends on hardware, load, cosmic rays). Statistical simulation with confidence intervals is the ceiling.
- **Custom logic** — depends on whether you constrain it to a verifiable subset or allow arbitrary code

### Session Types: The Missing Piece Nobody Talks About

**Multiparty Session Types (MPST)** can formally guarantee that distributed implementations conform to their communication protocols — at compile time. The spec defines a "global type" (the protocol), which is projected into per-participant "local types," and the type system ensures each participant's code follows its local type. This gives you:

- Deadlock freedom (proved)
- Protocol conformance (proved)
- No unexpected messages (proved)

Tools like [Scribble](https://dl.acm.org/doi/10.1145/3586031) already generate protocol-conforming Scala/Rust code from session type specs. Combining session types with simulation would let you **prove protocol correctness AND simulate timing behavior** — something no existing tool does.

### So the real framing is:

Not "accept a spectrum of weak assurance" but rather:

```
PROVE:   safety + liveness + protocol shape + deployment invariants
BOUND:   timing/performance via statistical simulation
GENERATE: protocol skeletons correct-by-construction
MONITOR:  custom logic + production drift via runtime verification
```

This is dramatically stronger than any existing approach. TLA+ only proves safety/liveness. P only generates code. FoundationDB only tests. Your approach would cover ALL of these through ONE specification.

---

## Proposed Architecture

```
                    ┌─────────────────────────────┐
                    │     SPECIFICATION LAYER      │
                    │  (Simulation DSL / Language)  │
                    │                               │
                    │  - Entity behaviors           │
                    │  - Network topology & model   │
                    │  - Failure model              │
                    │  - Timing constraints          │
                    │  - Deployment topology         │
                    │  - Correctness properties      │
                    │  - SLO definitions             │
                    └───┬────────┬────────┬────────┘
                        │        │        │
              ┌─────────▼┐ ┌────▼─────┐ ┌▼──────────┐
              │  EXTRACT  │ │ SIMULATE │ │ GENERATE  │
              │  to model │ │ (DES     │ │ impl code │
              │  checker  │ │  engine) │ │ + deploy  │
              └─────┬─────┘ └────┬─────┘ └─────┬────┘
                    │            │              │
              ┌─────▼─────┐ ┌───▼──────┐  ┌───▼────────┐
              │ Safety &   │ │ Bug      │  │ Production │
              │ liveness   │ │ finding  │  │ system     │
              │ proofs     │ │ + perf   │  │            │
              └────────────┘ │ analysis │  └───┬────────┘
                             └──────────┘      │
                                          ┌────▼────────────┐
                                          │ RUNTIME MONITOR │
                                          │ (conformance +  │
                                          │  drift detect)  │
                                          └─────────────────┘
```

### The Specification Language

This is the core innovation. It needs to be:

- **Formal enough** to extract a model-checkable subset (safety/liveness properties → TLA+ or similar)
- **Rich enough** to express timing, hardware, deployment as first-class concerns
- **Executable** as a discrete-event simulation
- **Restrictive enough** in its protocol layer that code generation is feasible

Think of it as **three sub-languages**:

1. **Protocol layer** (extractable to model checker): State machines, message types, invariants. Restricted — no arbitrary computation. This is what gets formally verified.

2. **Environment layer** (simulation-only): Network models, failure models, hardware models, workload models. This doesn't get verified formally — it gets simulated statistically.

3. **Deployment layer** (novel): Topology, rollout strategy, health checks, rollback triggers, canary criteria. This gets both simulated AND can generate deployment automation (Kubernetes manifests, Terraform, etc.).

### The Bridge (Implementation ↔ Simulation)

This is the hardest part. Three complementary mechanisms:

**A. Code Generation (spec → impl)**
- Protocol state machines → Rust/Go implementation skeletons
- Message types → protobuf/gRPC definitions
- Deployment topology → infrastructure-as-code
- This is "correct by construction" for the generated parts

**B. Dual Execution (FoundationDB-style)**
- Real implementation code plugs into the simulation engine
- Simulation replaces OS/network layer (injectable I/O)
- Same binary runs in simulation and production
- This validates that CUSTOM logic (not generated) behaves correctly

**C. Trace Conformance (production → spec)**
- Real system emits structured event traces
- Traces are replayed through the simulation
- Divergences → specification drift alerts
- This is the ongoing assurance layer

### What Makes This Different From Everything Else

The **deployment-as-specification** aspect is genuinely novel. No existing tool models:

- "During a rolling update from v2 to v3, the system must maintain read availability"
- "Canary deployment routes 5% of traffic to new version; abort if error rate > 1%"
- "Blue/green cutover completes in < 30s with zero dropped requests"

These are currently verified by... running them and hoping. A simulation that models the deployment process itself — nodes going down, new versions starting, traffic shifting — would let you **verify deployment safety before deploying**.

---

---

# Part 2: Deep Design Exploration

---

## The Specification Language: Concrete Design

### The Core Tension

The language must be:
- **Restricted enough** for formal verification (no Turing-complete protocol logic)
- **Expressive enough** to model real distributed systems
- **Executable** as a simulation
- **Compilable** to implementation code

Resolution: **a stratified language with three tiers**, each with different expressiveness/verifiability tradeoffs.

### Tier 1: Protocol Core (Formally Verifiable)

This tier defines the **logical heart** of the distributed system. It is:
- Restricted to guarded commands, finite state, bounded data structures
- Extractable to TLA+ for model checking
- Extractable to session type projections for conformance checking
- Code-generatable to Rust/Go

```
// ─── Message Types ──────────────────────────────────────
message RequestVote {
  term: int
  candidate_id: node_id
  last_log_index: int
  last_log_term: int
}

message VoteResponse {
  term: int
  vote_granted: bool
}

message AppendEntries {
  term: int
  leader_id: node_id
  prev_log_index: int
  prev_log_term: int
  entries: list[LogEntry]
  leader_commit: int
}

// ─── Global Protocol (Session Type) ────────────────────
// This is the KEY innovation: session types define the
// SHAPE of communication. The model checker verifies
// the LOGIC. Together they cover protocol correctness.

protocol RaftElection {
  // Candidate broadcasts vote requests, collects responses
  Candidate -> *Follower: RequestVote
  *Follower -> Candidate: VoteResponse
}

protocol RaftReplication {
  // Leader sends entries, followers acknowledge
  rec Loop {
    Leader -> *Follower: AppendEntries
    *Follower -> Leader: AppendEntriesResponse
    continue Loop
  }
}

// ─── Role State Machines ───────────────────────────────
role Follower {
  state {
    current_term: int = 0
    voted_for: node_id? = none
    log: list[LogEntry] = []
    commit_index: int = 0
  }

  on RequestVote(msg) from candidate {
    if msg.term > self.current_term {
      self.current_term = msg.term
      self.voted_for = none
    }

    let grant = msg.term >= self.current_term
             && (self.voted_for == none || self.voted_for == candidate)
             && msg.last_log_term >= self.log.last_term
             && msg.last_log_index >= self.log.last_index

    if grant { self.voted_for = candidate }

    reply VoteResponse { term: self.current_term, vote_granted: grant }
  }

  on ElectionTimeout {
    transition Candidate
  }
}

role Candidate {
  state { votes_received: set[node_id] = {self} }

  on enter {
    self.current_term += 1
    self.voted_for = self
    broadcast RequestVote { ... }
    start_timer ElectionTimeout(random(150ms, 300ms))
  }

  on VoteResponse(msg) from voter {
    if msg.vote_granted { self.votes_received.add(voter) }
    if self.votes_received.size >= quorum {
      transition Leader
    }
  }
}

role Leader { ... }

// ─── Safety Invariants (extracted to TLA+) ─────────────
invariant OneLeaderPerTerm {
  forall n1, n2 in nodes:
    n1.role == Leader && n2.role == Leader
    => n1 == n2 || n1.current_term != n2.current_term
}

invariant LogMatching {
  forall n1, n2 in nodes:
    forall i in 1..min(n1.log.length, n2.log.length):
      n1.log[i].term == n2.log[i].term
      => n1.log[1..i] == n2.log[1..i]
}

// ─── Liveness (needs fairness assumption) ──────────────
liveness EventualLeader {
  // Under fair scheduling: eventually a leader is elected
  assuming fair_scheduler
  eventually exists n in nodes: n.role == Leader
}
```

**What happens with this tier:**
1. **Session types** (`protocol RaftElection { ... }`) are projected per-role. Generated code is type-checked: a Follower cannot send an `AppendEntries` message (only Leaders can). This is a **compile-time guarantee** — not a test.
2. **Invariants** are extracted to TLA+ and model-checked exhaustively for small N (e.g., 3-5 nodes). This proves safety.
3. **Liveness** is checked via TLA+ temporal logic with fairness assumptions.
4. **Role state machines** generate implementation skeletons in the target language.

### Tier 2: Environment Model (Simulated, Not Proven)

This tier wraps the protocol in realistic physical conditions. It's expressed in the same spec file but explicitly marked as non-verifiable — it's simulated statistically.

```
// ─── Network Model ────────────────────────────────────
environment Production {
  network {
    // Within a datacenter
    intra_dc_latency: normal(mean=0.5ms, stddev=0.2ms)
    intra_dc_loss: 0.0001  // 0.01%

    // Cross-datacenter
    cross_dc_latency: normal(mean=50ms, stddev=10ms)
    cross_dc_loss: 0.001   // 0.1%

    // Partition model
    partitions {
      mtbf: 720h             // mean time between failures
      duration: normal(5min, 2min)
      scope: random_bisection // how nodes are split
    }
  }

  // ─── Hardware Model ──────────────────────────────────
  hardware {
    disk {
      write_latency: normal(0.1ms, 0.05ms)
      fsync_latency: normal(2ms, 1ms)
      iops: 50000
      failure_rate: 0.001 / year
    }
    cpu {
      cores: 8
      gc_pause: occasional(mean=10ms, frequency=0.1/s)
    }
  }

  // ─── Workload ────────────────────────────────────────
  workload {
    client_rate: poisson(1000/s)
    payload_size: uniform(100B, 10KB)
    read_write_ratio: 0.8
    think_time: exponential(mean=50ms)
  }

  // ─── SLO Definitions ────────────────────────────────
  slo {
    read_latency_p99 < 10ms
    write_latency_p99 < 50ms
    availability > 99.9%
    recovery_time < 10s  // after single node failure
  }
}
```

**What happens with this tier:**
1. **Compiled to simulation**: Network model → `NetworkLink` entities with appropriate latency distributions. Hardware model → disk/CPU entities with timing. Workload → `Source` entities.
2. **SLOs become assertions**: After simulation, check if SLOs were met. Run 1000 simulations with different seeds → get confidence intervals.
3. **NOT formally verified**: You can't prove p99 latency < 10ms because it depends on hardware. But you can show it holds in 99.7% of simulations under realistic conditions.

### Tier 3: Deployment Model (Model-Checkable + Simulatable)

This is the novel tier. Deployment processes are **finite-state** (each node is in one of ~5 states during a rollout), so safety properties CAN be model-checked.

```
// ─── Versioning ────────────────────────────────────────
version v2 {
  // Defines the protocol spec for this version
  uses RaftProtocol_v2
}

version v3 {
  uses RaftProtocol_v3

  // Compatibility declaration (critical for mixed-version safety)
  compatible_with v2 {
    // v3 nodes can receive v2 messages (backward compatible)
    // v2 nodes can receive v3 messages IF new_field is optional
    new_field "priority" in AppendEntries is optional
  }
}

// ─── Deployment Strategy ──────────────────────────────
deployment RollingUpdate(from: v2, to: v3) {
  topology {
    nodes: 5
    regions: [us-east(3), us-west(2)]
  }

  strategy {
    order: one_region_at_a_time  // or: one_at_a_time, all_at_once
    batch_size: 1
    pause_between: 30s

    per_node {
      drain_connections(timeout: 10s)
      stop_process
      upgrade_binary(to: v3)
      start_process
      health_check(endpoint: /health, timeout: 5s, retries: 3)
      restore_connections
    }
  }

  // ─── Deployment Invariants (MODEL-CHECKED) ─────────
  invariant AvailableDuringRollout {
    // At every intermediate state:
    count(nodes where state == Running) >= 3  // quorum
  }

  invariant NoSplitBrain {
    // Mixed versions must not form separate quorums
    forall partition in possible_partitions:
      at_most_one_quorum(partition)
  }

  invariant DataSafety {
    // Committed entries are never lost
    forall entry in committed_log:
      exists n in running_nodes: entry in n.log
  }

  // ─── Rollback Triggers (SIMULATED) ─────────────────
  rollback_trigger {
    error_rate > 1%
    latency_p99 > 2x baseline
    any node fails health_check 3 times
  }

  // ─── Generates ──────────────────────────────────────
  generates {
    kubernetes_manifest   // K8s Deployment + rolling update config
    rollout_controller    // Custom controller enforcing invariants
    runbook               // Human-readable deployment procedure
  }
}
```

**What happens with this tier:**
1. **Deployment invariants are model-checked**: Enumerate all possible orderings of node upgrades. For each intermediate state, verify that `AvailableDuringRollout`, `NoSplitBrain`, `DataSafety` hold. This is exhaustive for small cluster sizes.
2. **Rollback triggers are simulated**: Run the deployment simulation with injected failures (node crash during upgrade, slow health check, etc.). Verify rollback fires correctly.
3. **Deployment automation is generated**: K8s manifests, Argo Rollout specs, or custom deployment scripts — directly from the spec.
4. **Mixed-version compatibility is verified**: The `compatible_with` declaration is checked against the protocol diff between versions. If v3 adds a required field that v2 doesn't understand, the model checker flags it BEFORE you deploy.

---

## Session Types + Simulation: The Technical Integration

This is the key novel combination. Here's how they work together concretely.

### What Session Types Give You (Compile-Time)

A global protocol type is **projected** to per-role local types:

```
// Global protocol
protocol RaftElection {
  Candidate -> *Follower: RequestVote
  *Follower -> Candidate: VoteResponse
}

// Projected local type for Candidate:
//   send RequestVote to each Follower
//   receive VoteResponse from each Follower

// Projected local type for Follower:
//   receive RequestVote from Candidate
//   send VoteResponse to Candidate
```

Generated code for each role includes a **session channel** that enforces the protocol:

```rust
// Generated Rust (conceptual)
impl CandidateSession {
    // Type system enforces: you MUST send RequestVote before
    // you can receive VoteResponse. You CANNOT send AppendEntries
    // from a Candidate session.

    fn run(self, channel: CandidateChannel<SendRequestVote>) {
        // After sending, channel type changes to ReceiveVoteResponse
        let channel = channel.send_to_all(request_vote);

        // Now we can only receive VoteResponses
        for response in channel.receive_all() {
            // ...
        }
        // channel is consumed — can't reuse
    }
}
```

**This eliminates an entire category of bugs at compile time:**
- Sending wrong message type → compile error
- Sending in wrong order → compile error
- Forgetting to handle a response → compile error
- Deadlock from mismatched send/receive → compile error

### What Simulation Gives You (Run-Time)

Session types guarantee the **shape** of communication but NOT:
- Whether the vote logic is correct (could grant votes to wrong candidates)
- Whether log replication converges
- Whether the system performs under load
- Whether network partitions are handled correctly

The simulation covers these:

```python
# Simulation test derived from spec
def test_raft_safety_under_partition():
    """The spec's invariants must hold even during network partitions."""
    sim = compile_spec_to_simulation("raft.spec",
        environment="Production",
        nodes=5)

    # Inject partition at random time
    sim.schedule_fault(
        time=random.uniform(5, 30),
        fault=NetworkPartition(
            group_a=random.sample(nodes, 2),
            group_b=remaining(nodes)
        ),
        duration=random.uniform(1, 60)
    )

    # Run and check spec invariants continuously
    summary = sim.run(duration=120,
        check_invariants=["OneLeaderPerTerm", "LogMatching"],
        check_slos=["read_latency_p99", "availability"])

    assert summary.invariants_violated == []
    assert summary.slo_met["availability"]
```

### The Combined Guarantee

```
Session Types:  "Implementation follows the protocol shape"       (PROVED)
Model Checker:  "Protocol logic satisfies safety invariants"      (PROVED)
Simulation:     "System meets SLOs under realistic conditions"    (BOUNDED)
Together:       "A correct, performant, well-specified system"
```

No existing tool gives you more than one of these. The combination is the innovation.

---

## The Bridge: Technical Deep Dive

The bridge is the mechanism that connects the specification to the implementation. There are three complementary approaches, and a real system would use all three.

### Bridge A: Code Generation (Correct by Construction)

**What gets generated from the spec:**

| Spec Element | Generated Artifact | Correctness |
|---|---|---|
| `message RequestVote { ... }` | Protobuf/FlatBuffers schema + serde code | Exact (structural) |
| `protocol RaftElection { ... }` | Session-typed channel interfaces | Proved (session types) |
| `role Follower { state { ... } }` | State struct with accessors | Exact (structural) |
| `on RequestVote(msg) { ... }` | Message handler skeleton | Partial (logic is generated, but custom logic needs human review OR LLM + verification) |
| `invariant OneLeaderPerTerm` | Runtime assertion + model checker input | Dual-use |
| `deployment RollingUpdate` | K8s manifest + rollout controller | Structural |

**What is NOT generated (the escape hatches):**
- Storage layer (how to persist the log — RocksDB? SQLite? custom?)
- Application-level state machine (what commands mean to the application)
- Custom metrics and monitoring
- Integration with existing infrastructure

Escape hatches have **contracts** derived from the spec:

```rust
// Generated interface with spec-derived contracts
trait LogStorage {
    /// Pre: index > 0
    /// Post: returns entry at index, or None if not present
    /// Invariant: if append(entry) was called with index i,
    ///            then get(i) returns that entry until truncate
    fn get(&self, index: usize) -> Option<LogEntry>;

    /// Pre: entry.index == self.last_index() + 1
    /// Post: self.last_index() == entry.index
    fn append(&mut self, entry: LogEntry);

    /// Pre: from_index > 0
    /// Post: self.last_index() == from_index - 1
    fn truncate_from(&mut self, from_index: usize);
}

// The LLM (or human) implements this trait.
// Contracts are checked at test time via property-based testing.
// In production, contracts become runtime assertions.
```

### Bridge B: Dual Execution (Same Code, Two Runtimes)

This is the FoundationDB approach: the implementation code runs in BOTH simulation and production. The simulation replaces the I/O layer.

```rust
// The implementation uses an abstract I/O trait
trait NetworkIO {
    async fn send(&self, to: NodeId, msg: Message) -> Result<()>;
    async fn recv(&self) -> Result<(NodeId, Message)>;
}

trait ClockIO {
    fn now(&self) -> Instant;
    async fn sleep(&self, duration: Duration);
}

trait DiskIO {
    async fn write(&self, data: &[u8]) -> Result<()>;
    async fn fsync(&self) -> Result<()>;
    async fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>>;
}

// Production runtime: real TCP sockets, real clock, real disk
struct ProductionRuntime { ... }

// Simulation runtime: routes through the simulation engine
struct SimulationRuntime {
    // Backed by happysimulator's Network, NodeClock, etc.
    sim_network: SimNetwork,
    sim_clock: SimClock,
    sim_disk: SimDisk,
}
```

**The critical advantage**: The SAME binary (with different runtime injection) runs in both environments. No gap between "what we tested" and "what we deployed."

**How this integrates with the spec:**
1. Spec defines the protocol → generates the handler skeletons
2. Developer fills in business logic using the I/O traits
3. Spec's environment model configures the `SimulationRuntime`
4. Same code runs with `ProductionRuntime` in prod

### Bridge C: Trace Conformance (Runtime Drift Detection)

The production system is instrumented to emit **structured traces** that can be compared against the simulation.

```
// Trace event format (emitted by both simulation and production)
{
  "timestamp": "2024-01-15T10:30:00.123456Z",
  "node": "node-2",
  "event_type": "state_transition",
  "from_role": "Follower",
  "to_role": "Candidate",
  "trigger": "ElectionTimeout",
  "state_snapshot": {
    "current_term": 5,
    "voted_for": "node-2",
    "log_length": 42,
    "commit_index": 40
  }
}
```

**Conformance checking works at three levels:**

1. **Structural conformance**: Does the sequence of message types match the session type? (e.g., did we ever see a Follower send an AppendEntries?) — This should NEVER fail if session types are enforced at compile time.

2. **Behavioral conformance**: Given the same sequence of inputs, does the implementation produce the same state transitions as the simulation? — Feed production traces into simulation, compare outputs.

3. **Statistical conformance**: Do the observed latency distributions match the simulation's predictions within tolerance? — Compare production p99 against simulation p99. Alert on sustained divergence.

**Drift detection in practice:**

```python
# Continuous conformance checker (runs as sidecar or batch job)
def check_conformance(production_traces, spec):
    sim = compile_spec_to_simulation(spec)

    for trace in production_traces:
        # Replay the exact same sequence of external events
        sim.replay(trace.external_events)

        # Compare state transitions
        for prod_transition, sim_transition in zip(
            trace.state_transitions, sim.state_transitions
        ):
            if prod_transition.to_state != sim_transition.to_state:
                alert(DriftDetected(
                    node=trace.node,
                    expected=sim_transition,
                    actual=prod_transition,
                    context=trace.surrounding_events
                ))

    # Statistical comparison
    prod_latencies = trace.latency_distribution()
    sim_latencies = sim.latency_distribution()
    if ks_test(prod_latencies, sim_latencies).p_value < 0.01:
        alert(PerformanceDrift(
            metric="latency",
            production=prod_latencies.describe(),
            simulation=sim_latencies.describe()
        ))
```

---

## Deployment-as-Specification: Detailed Design

This is the most novel aspect. Let me expand on how deployment becomes a first-class verifiable concern.

### Why Deployment Is Finite-State (and Thus Model-Checkable)

During a rolling update of an N-node cluster, each node is in one of these states:

```
NodeDeployState = Running_V1 | Draining | Stopped | Starting_V2 | Running_V2
```

For N=5 nodes, there are 5^5 = 3125 possible global states. Most are unreachable given the rollout ordering constraints. The reachable state space is typically ~50-200 states — easily model-checked.

For each reachable state, you verify:
- **Quorum availability**: Are enough nodes in `Running_*` state to form a quorum?
- **Data safety**: Is every committed log entry present on at least one `Running_*` node?
- **Version compatibility**: Can `Running_V1` and `Running_V2` nodes communicate?
- **Progress**: Does the rollout eventually complete (liveness)?

### The Deployment State Machine

```
// Each node follows this state machine during rollout:
//
//  Running_V1 ──drain──> Draining ──stop──> Stopped
//       │                                      │
//       │                                   upgrade
//       │                                      │
//       │                                      v
//       └──────────────────────────────── Starting_V2 ──health_ok──> Running_V2
//                   (skip drain if                         │
//                    not receiving                     health_fail
//                    traffic)                              │
//                                                          v
//                                                     ROLLBACK
//                                                    (all nodes → V1)
```

### Simulating the Deployment Process

The simulation models the deployment AS part of the system simulation:

```
// The deployment process itself is an entity in the simulation
entity RolloutController {
  state {
    plan: list[NodeId]          // order to upgrade
    current_batch: int = 0
    rollback_triggered: bool = false
  }

  on StartRollout {
    for node in plan[current_batch] {
      send DrainNode to load_balancer for node
      schedule CheckDrained(node) after 10s
    }
  }

  on CheckDrained(node) {
    if node.active_connections == 0 {
      send StopProcess to node
      schedule UpgradeNode(node) after 2s  // binary swap
    } else if elapsed > drain_timeout {
      send ForceStop to node  // forceful drain
    }
  }

  on HealthCheckPassed(node) {
    send RestoreTraffic(node) to load_balancer
    current_batch += 1
    if current_batch < plan.length {
      schedule StartNextBatch after pause_between
    }
  }

  on HealthCheckFailed(node) {
    rollback_triggered = true
    for all_nodes { send Rollback(to: v1) }
  }
}
```

This lets you simulate:
- What happens if a node crashes DURING the upgrade?
- What happens if health checks are slow?
- What happens if the load balancer takes time to drain?
- What happens if you need to rollback mid-deployment?

### Mixed-Version Verification

The most subtle deployment bugs come from **mixed-version communication**. The spec language makes this explicit:

```
version v3 {
  compatible_with v2 {
    // v3 adds a new field to AppendEntries
    new_field "priority" in AppendEntries {
      // v2 nodes will ignore this field (backward compat)
      // v3 nodes will default it to 0 if not present (forward compat)
      default: 0
      required: false
    }

    // v3 changes the election timeout range
    changed "election_timeout" {
      v2: random(150ms, 300ms)
      v3: random(200ms, 400ms)
      // This is SAFE: overlapping ranges mean mixed clusters
      // can still elect leaders
    }
  }
}
```

The model checker verifies: "In a cluster with some V2 nodes and some V3 nodes, do the protocol invariants still hold?" This catches the class of bugs where a new version changes behavior in a way that's safe in a homogeneous cluster but breaks in a mixed-version cluster.

---

## Novelty Assessment: What's New Here vs. What's Incremental

| Aspect | Novelty | Why |
|--------|---------|-----|
| Simulation as spec | **Medium** | FoundationDB pioneered this, P language does it partially. The idea is known. |
| Session types + simulation | **High** | Nobody combines compile-time protocol proofs with runtime simulation. This is genuinely new. |
| Deployment-as-specification | **Very High** | No existing formal methods tool models deployment processes. This is a white space. |
| Unified spec → multiple backends | **High** | One spec feeding model checker + simulator + code gen + runtime monitor is novel. |
| Timing properties in formal spec | **Medium** | Statistical model checking exists but isn't integrated with protocol verification. |
| Runtime conformance via trace replay | **Medium** | Digital twins do this for cyber-physical systems; applying it to distributed software is newer. |

**The strongest novel contribution would be: a specification language where session types guarantee protocol correctness, model checking guarantees safety/liveness, simulation validates timing/performance, AND the deployment process itself is specified and verified.** No one tool does more than two of these today.

---

## The Core Use Case: AI as Implementor, Spec as Guardrail

The primary use case is NOT "human writes spec and implementation separately, hopes they match." It is:

**The spec is a fixed point. An AI writes arbitrary low-level implementation code. The verification stack tells the AI whether that code is correct. The AI iterates rapidly until it converges on a correct, high-performance implementation.**

This reframes the entire design. Code generation is secondary. The primary outputs of the spec are:

1. **The I/O Interface** — the contract the implementation must satisfy
2. **The Oracle** — the simulation of correct behavior
3. **The Verification Stack** — tiered checks from compile-time to statistical

### Why This Matters Now

AI coding agents are good at writing low-level systems code — Rust, C, custom allocators, io_uring, lock-free data structures. What they're NOT good at is reasoning about distributed systems correctness. The spec handles the reasoning; the AI handles the implementation.

```
HUMAN INTENT                    AI IMPLEMENTATION
"Build a Raft cluster           "Here's the Rust code with
 that survives network           io_uring for disk, custom
 partitions, handles             arena allocator for log
 rolling updates, and            entries, SIMD for scanning
 meets p99 < 10ms"              match indices, and zero-copy
                                 message parsing"
        │                               │
        ▼                               ▼
┌──────────────┐              ┌──────────────────┐
│     SPEC     │              │  IMPLEMENTATION   │
│  (behavioral │◄────────────►│  (any style,      │
│   contract)  │  I/O boundary│   close to metal) │
└──────┬───────┘              └────────┬─────────┘
       │                               │
       ▼                               ▼
┌──────────────┐              ┌──────────────────┐
│   ORACLE     │   compare    │   ACTUAL          │
│  (simulation │◄────────────►│   BEHAVIOR        │
│   output)    │              │   (in simulation) │
└──────────────┘              └──────────────────┘
       │
       ▼
   PASS / FAIL + rich diagnostics
       │
       ▼
   AI iterates on implementation
```

### The Spec Produces Three Artifacts (Not Code)

**1. The I/O Interface**

This is the ONLY coupling point between spec and implementation. The AI's code can do anything internally — unsafe Rust, SIMD, raw pointers, custom syscalls — but it talks to the outside world through these interfaces:

```rust
// Generated from spec — this is the CONTRACT
trait RaftIO: Send + Sync {
    // Network (all messages go through here)
    async fn send_message(&self, to: NodeId, msg: RaftMessage) -> Result<()>;
    async fn receive_message(&self) -> Result<(NodeId, RaftMessage)>;

    // Persistent storage (all durability goes through here)
    async fn persist_state(&self, state: &PersistentState) -> Result<()>;
    async fn append_log(&self, entries: &[LogEntry]) -> Result<()>;
    async fn truncate_log(&self, from_index: u64) -> Result<()>;

    // Time (all scheduling goes through here)
    fn now(&self) -> Instant;
    async fn sleep_until(&self, deadline: Instant);
    fn random_duration(&self, min: Duration, max: Duration) -> Duration;
}

// The AI implements:
struct RaftNode<IO: RaftIO> {
    io: IO,
    // ... whatever internal data structures the AI chooses
    // Arena-allocated log? Fine.
    // Lock-free message queue? Fine.
    // SIMD-accelerated commit index scan? Fine.
    // Raw pointer manipulation? Fine (if it satisfies the behavioral contract).
}
```

The I/O boundary is at the **protocol level**: send/receive messages, persist/read state, read clock. This gives the AI FULL freedom for internal implementation while keeping all externally-visible behavior under the spec's control.

**Too high** (e.g., `process_request() -> Response`) → can't verify protocol behavior.
**Too low** (e.g., `syscall(SYS_write, fd, buf, len)`) → can't write efficient code.
**Just right** (e.g., `send_message`, `persist_state`, `now`) → full internal freedom, full behavioral verification.

**2. The Oracle (Simulation as Ground Truth)**

The spec, running as a simulation, defines correct behavior. For any sequence of inputs and failures, the oracle says what SHOULD happen:

```python
# The oracle is the spec compiled to a simulation
oracle = compile_spec_to_simulation("raft.spec", nodes=5)

# Given these inputs:
oracle.inject_event(ClientWrite("key1", "value1"), at=1.0)
oracle.inject_event(ClientWrite("key2", "value2"), at=2.0)
oracle.inject_fault(Partition([0,1], [2,3,4]), at=5.0, duration=30.0)
oracle.inject_event(ClientWrite("key3", "value3"), at=10.0)

# The oracle produces the CORRECT behavior:
oracle.run()
# → leader elected by t=3.0
# → key1, key2 committed by t=4.0
# → after partition: new leader elected in majority partition
# → key3 committed by new leader
# → after heal: old leader steps down, catches up
# → ALL invariants hold throughout
```

The AI's implementation is run through the SAME sequence. If it produces different behavior → bug. If it produces the same behavior → correct (for this scenario). Run 1000 scenarios → high confidence.

**3. The Verification Stack (Tiered for Speed)**

The AI iterates RAPIDLY. This means verification must be FAST for most checks, with slower exhaustive checks reserved for final validation:

```
Tier 0: COMPILE (milliseconds)
  Session types: Does the code compile against the protocol interface?
  Type check: Are message types correct?
  → Catches: wrong message types, protocol shape violations, deadlock potential

Tier 1: UNIT CONTRACTS (seconds)
  Property-based tests on individual handlers:
    "For any valid AppendEntries message, the response term >= message term"
    "persist_state is called before sending VoteResponse"
  → Catches: contract violations, missed edge cases

Tier 2: TARGETED SCENARIOS (seconds)
  Run specific failure patterns against the oracle:
    "3-node cluster, leader crash at t=5"
    "5-node cluster, partition at t=10, heal at t=40"
    "5-node cluster, 2 simultaneous crashes"
  → Catches: behavioral bugs, incorrect state transitions

Tier 3: STATISTICAL SUITE (minutes)
  Run 1000 seeds with randomized faults:
    Random partitions, crashes, message delays, clock skew
  Check ALL invariants + ALL SLOs
  → Catches: rare race conditions, performance regressions

Tier 4: MODEL CHECK (minutes-hours)
  Extract protocol logic to TLA+, exhaustive state exploration
  → Catches: ANY safety violation for bounded parameters (PROOF)
```

The AI runs Tier 0-2 on every change (takes <10 seconds). Tier 3 after a batch of changes. Tier 4 when the protocol logic changes.

### The Rapid Iteration Loop

```
AI reads spec → understands behavioral contract
    │
    ▼
AI writes implementation (Rust, close to the metal)
    │
    ▼
Tier 0: Compile ─── FAIL → AI fixes type errors ──┐
    │ PASS                                          │
    ▼                                               │
Tier 1: Contracts ─── FAIL → AI reads:             │
    │ PASS              "persist_state not called   │
    ▼                    before VoteResponse at     │
                         line 247" ─────────────────┤
Tier 2: Scenarios ─── FAIL → AI reads:             │
    │ PASS              "OneLeaderPerTerm violated  │
    ▼                    at t=15.3s. node-0 and     │
                         node-2 both Leader in      │
Tier 3: Stats ─────── FAIL → AI reads:     term 5. │
    │ PASS              "p99 latency 47ms,  Event   │
    ▼                    SLO requires <10ms. trace:  │
                         Bottleneck: fsync  [...]"──┘
VALIDATED ✓                                 │
    │                                       │
    ▼                            ◄──────────┘
Ship to production               AI modifies and
(with runtime monitoring)        re-runs verification
```

### Rich Error Feedback (Critical for AI Iteration)

When verification fails, the AI needs ACTIONABLE diagnostics, not just "test failed":

```
INVARIANT VIOLATION: OneLeaderPerTerm
  Time: t=15.3s
  Violated by: node-0 (Leader, term=5) AND node-2 (Leader, term=5)

  Event trace leading to violation:
    t=0.0   All nodes start as Follower
    t=1.7   node-0 election timeout → becomes Candidate (term=1)
    t=1.8   node-0 wins election → becomes Leader (term=1)
    t=5.0   PARTITION: {node-0, node-1} | {node-2, node-3, node-4}
    t=7.2   node-2 election timeout → becomes Candidate (term=2)
    t=7.4   node-2 wins election in majority → becomes Leader (term=2)
    t=10.0  PARTITION HEALS
    t=10.1  node-0 receives AppendEntries from node-2 (term=2)
    t=10.1  BUG: node-0 does NOT step down despite seeing higher term
    t=15.3  node-0 sends AppendEntries (term=5) ← term incremented by
            election attempts while partitioned, but never stepped down

  Root cause: handle_append_entries() missing term check.
  Expected behavior (from oracle): node-0 steps down at t=10.1

  Relevant spec section:
    role Leader {
      on AppendEntries(msg) {
        if msg.term > self.current_term {
          self.current_term = msg.term
          transition Follower    ← THIS TRANSITION IS MISSING
        }
      }
    }
```

This level of diagnostic detail lets the AI pinpoint and fix the bug in a single iteration.

### Why "Close to the Metal" Is Compatible with Formal Verification

The spec constrains WHAT (behavioral correctness). The AI controls HOW (implementation strategy). These are orthogonal:

| What the AI might do internally | Does the spec care? | Why |
|---|---|---|
| Custom arena allocator for log entries | No | Internal memory management doesn't change observable behavior |
| Lock-free MPSC queue for message dispatch | No | Internal concurrency strategy doesn't change protocol messages |
| io_uring for batched disk writes | **Partially** | Must still satisfy `persist_state` contract (data durable after call returns) |
| SIMD for scanning match_index array | No | Internal computation strategy doesn't change results |
| unsafe Rust with raw pointer manipulation | **Partially** | Memory corruption could produce wrong messages (caught by simulation) |
| Zero-copy message parsing | No | Parsing strategy doesn't change message semantics |
| Custom TCP stack | **Partially** | Must still satisfy `send_message` / `receive_message` contracts |

The "partially" cases are caught by the simulation: if io_uring doesn't actually fsync when it should, the simulation will detect data loss after a simulated crash. If unsafe code corrupts memory, the simulation will see incorrect messages or state transitions.

For low-level bugs that the behavioral simulation can't catch (memory leaks, undefined behavior, data races), complement with:
- **Sanitizers** (ASan, MSan, TSan) — run during Tier 2/3 simulation
- **Miri** (for Rust) — checks unsafe code for UB
- **Loom** (for Rust) — checks lock-free code for data races
- **Fuzzing** of the serialization layer

These are standard systems programming tools that the AI should run alongside the spec verification.

### The AI's Degrees of Freedom

Given a spec, the AI has full freedom over:

| Degree of Freedom | Examples |
|---|---|
| **Data structures** | B-tree log vs. LSM-tree vs. append-only file vs. arena-allocated array |
| **Concurrency model** | Async/await vs. threads vs. io_uring vs. green threads |
| **Memory management** | Custom allocators, memory pools, zero-copy buffers |
| **Serialization** | Protobuf vs. FlatBuffers vs. Cap'n Proto vs. raw bytes |
| **Network stack** | TCP vs. QUIC vs. DPDK vs. custom kernel bypass |
| **Disk I/O** | Buffered vs. direct I/O vs. io_uring vs. mmap |
| **Optimization level** | SIMD, branch prediction hints, cache-line alignment, prefetching |
| **Architecture** | Monolithic vs. microkernel vs. library-level components |

All of these are INTERNAL and invisible through the I/O boundary. The spec doesn't care. The simulation verifies that the BEHAVIOR is correct regardless of the implementation strategy.

This means the AI can aggressively optimize — try a completely different data structure, rewrite the hot path in unsafe code, switch from async to io_uring — and each time, the verification stack confirms correctness in seconds.

### The Specification as Stable Abstraction Layer

```
LAYER 3: Human Intent
  "Build a Raft-based distributed KV store that survives
   datacenter failures and meets p99 < 10ms"
         │
         ▼
LAYER 2: Specification (STABLE — changes rarely)
  Protocol behavior, safety invariants, liveness properties,
  deployment constraints, SLOs, environment model
         │
         ▼
LAYER 1: Implementation (VOLATILE — changes constantly)
  AI iterates rapidly on data structures, algorithms,
  optimizations, memory layout, concurrency strategy
         │
         ▼
LAYER 0: Hardware
  CPU, memory, disk, network — the metal
```

The spec is the **stable interface** between human intent (Layer 3) and machine implementation (Layer 1). Humans define WHAT at Layer 2. The AI implements HOW at Layer 1, iterating against the verification stack until all properties are satisfied.

This is the fundamental value proposition: **the spec makes AI-generated systems code trustworthy**, not by constraining how the AI writes code, but by comprehensively verifying what the code does.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Language design is wrong | High | Start with a deeply embedded DSL in Python, iterate before committing to custom syntax |
| Model checking doesn't scale | Medium | Use bounded model checking + statistical simulation (not exhaustive) |
| I/O boundary at wrong abstraction level | High | Start at protocol level (send_message, persist_state); adjust based on what the AI needs freedom over |
| Verification too slow for rapid iteration | High | Tiered approach: Tier 0-2 in seconds, Tier 3 in minutes, Tier 4 occasionally |
| Simulation doesn't catch low-level bugs | Medium | Complement with sanitizers, Miri, Loom, fuzzing during simulation runs |
| Oracle diverges from real-world behavior | Medium | Runtime conformance monitoring catches oracle inaccuracies; update environment model |
| Scope creep (trying to model everything) | High | Start with ONE protocol (Raft), ONE deployment model (rolling update), prove the concept |

---

---

# Part 3: Deep Technical Exploration

---

## The Spec Language: Formal Semantics

### Computational Model

Tier 1 is based on **communicating guarded-command state machines** (similar to Promela/SPIN, P language, or I/O Automata), enhanced with session types for communication structure. Formally:

```
System = (Nodes, Messages, Channels, Invariants)

Node = (State, Guards, Actions)
  State   = typed record (product of base + abstract types)
  Guard   = predicate over (local_state, incoming_message)
  Action  = state_update × message_sends × timer_ops

Transition = guard(state, msg) → (state', sends, timers)

Execution = interleaving of enabled transitions across all nodes
  (non-deterministic for model checking, scheduled for simulation)
```

### The Type System

**Base types** (finite for model checking):
```
int                  // bounded: model checker constrains range
bool
node_id              // drawn from finite set of participant names
enum { A | B | C }   // finite enumeration
```

**Compound types**:
```
record { field1: T1, field2: T2 }       // product type
list[T]                                  // ordered, bounded for model checking
set[T]                                   // unordered, bounded
map[K, V]                                // bounded
option[T]                                // T | none
```

**Abstract data types** (key innovation for bridging spec and impl):
```
// The spec reasons about abstract operations.
// The implementation provides concrete data structures.
// Refinement: concrete impl must satisfy abstract spec.

abstract type Log {
  operations:
    append(term: int, command: any) -> index: int
    get(index: int) -> option[LogEntry]
    truncate_from(index: int)
    entries_after(index: int) -> list[LogEntry]
    last_index -> int
    last_term -> int

  axioms:
    // After append, get returns the appended entry
    forall log, term, cmd:
      let idx = log.append(term, cmd)
      log.get(idx) == Some(LogEntry(idx, term, cmd))

    // Truncate removes everything from index onward
    forall log, idx:
      log.truncate_from(idx)
      log.last_index == idx - 1

    // Append is monotonic
    forall log, term, cmd:
      let old_idx = log.last_index
      log.append(term, cmd)
      log.last_index == old_idx + 1
}
```

**Session types** (for protocol shape):
```
// Based on Multiparty Session Types (MPST)
// Global type defines the interaction pattern

global type RaftReplication {
  rec Heartbeat {
    Leader -> Follower[i]: AppendEntries      // Leader sends to each follower
    Follower[i] -> Leader: AppendEntriesResponse
    continue Heartbeat
  }
}

// Projection to local types:
// Leader's local type:
//   rec { !AppendEntries[i].?AppendEntriesResponse[i].continue }
// Follower i's local type:
//   rec { ?AppendEntries.!AppendEntriesResponse.continue }

// The ! means "send", ? means "receive"
// These become type constraints on generated code
```

**Refinement annotations** (optional, for contract-based verification):
```
// Pre/post conditions on operations
on AppendEntries(msg) from leader {
  requires msg.term >= 0
  requires msg.prev_log_index >= 0

  // ... handler logic ...

  ensures self.current_term >= old(self.current_term)
  ensures reply.term == self.current_term
}
```

### What Tier 1 CANNOT Express (By Design)

These restrictions keep the language verifiable:

| Forbidden | Why | Workaround |
|-----------|-----|------------|
| Unbounded loops | State space explosion | Iterate over bounded collections; use `rec` for protocol loops |
| Recursive functions | Undecidable termination | Protocol recursion via `rec` in session types (structural, not general) |
| Arbitrary arithmetic | Undecidable in general | Bounded integers; model checker constrains range |
| Side effects beyond state/messages | Breaks formal model | I/O is in Tier 2 (environment) |
| Dynamic node creation | Infinite state space | Fixed participant set; dynamic membership is a protocol concern |
| Floating point | Non-deterministic on hardware | Time is handled in Tier 2; Tier 1 uses logical ordering |

### Parametric Model Checking

The spec is written without bounds. Model checking instantiates with small bounds:

```
// Spec says:
role Follower {
  state {
    log: Log    // unbounded in the spec
    current_term: int   // unbounded
  }
}

// Model checking annotation:
@model_check(
  nodes = [3, 5],            // check with 3 and 5 nodes
  max_term = 4,               // terms 0..4
  max_log_length = 5,         // log entries 0..5
  max_messages_in_flight = 10  // bound message buffers
)
invariant OneLeaderPerTerm { ... }
```

The model checker explores all states within these bounds. If an invariant holds for `(N=3, terms=4, log=5)`, it doesn't prove it for `(N=100, terms=1000000, log=∞)` — but experience with Raft/Paxos shows that most bugs appear at small parameter values. The simulation covers larger instances probabilistically.

### How Restrictions Map to the Simulation

The beautiful thing: **Tier 1 restrictions only constrain the model checker**. The simulation runs the FULL unrestricted version:

```
┌─────────────────────────────────────────────────────┐
│              Tier 1 Spec (restricted)                 │
│  "current_term: int" → model checks with term ≤ 4   │
│  "log: Log" → model checks with |log| ≤ 5           │
├─────────────────────────────────────────────────────┤
│       Model Checker          │    Simulation         │
│  Explores ALL interleavings  │  Explores MANY runs   │
│  with bounded parameters     │  with unbounded params │
│  PROVES for small instances  │  TESTS for real scale  │
└─────────────────────────────────────────────────────┘
```

---

## The Bridge: Technical Deep Dive

### Trace Conformance Algorithm

The key challenge: production traces are at a different level of abstraction than simulation events. A production trace has timestamps, raw bytes, OS-level details. The simulation works with typed messages and logical time.

**The Abstraction Function**

An abstraction function `α` maps production-level observations to spec-level events:

```
α: ProductionEvent → SpecEvent | Skip

Examples:
  α("TCP send 192.168.1.5:8080 → 192.168.1.6:8080 [bytes]")
    → SpecEvent(from="node-0", to="node-1", type="AppendEntries", payload=deserialize(bytes))

  α("GC pause 12ms")
    → Skip  (not relevant to protocol correctness)

  α("disk fsync completed in 2.3ms")
    → Skip for protocol conformance
    → TimingObservation(op="fsync", duration=2.3ms) for performance conformance
```

**Three Levels of Conformance Checking**

**Level 1: Protocol Trace Conformance (should never fail if session types compile)**

Check that the sequence of message types matches the session type:

```
Production trace (abstracted):
  node-0 → node-1: RequestVote
  node-1 → node-0: VoteResponse
  node-0 → node-1: AppendEntries
  node-1 → node-0: AppendEntriesResponse

Expected by session type:
  Candidate → Follower: RequestVote
  Follower → Candidate: VoteResponse
  Leader → Follower: AppendEntries
  Follower → Leader: AppendEntriesResponse

Result: MATCH ✓
```

If session types are enforced at compile time, Level 1 violations should be impossible. But checking it in production catches bugs from:
- Using an old binary that wasn't compiled with the session type checker
- Interacting with external systems not under session type control
- Cosmic rays (not joking — hardware faults can corrupt state)

**Level 2: State Conformance (the core drift detector)**

```python
def check_state_conformance(production_trace, spec):
    """
    For each state-changing event in the production trace,
    verify that the simulation reaches the same state.
    """
    sim = spec.instantiate_simulation()

    for event in production_trace.state_changing_events():
        # Feed the same external input to the simulation
        sim.inject_event(event.as_external_input())

        # Let sim process it
        sim.step()

        # Compare states (only Tier 1 state fields)
        prod_state = event.state_after
        sim_state = sim.get_node_state(event.node)

        divergences = []
        for field in spec.tier1_state_fields():
            if prod_state[field] != sim_state[field]:
                divergences.append(Divergence(
                    field=field,
                    production=prod_state[field],
                    simulation=sim_state[field],
                    cause=classify_divergence(event, field)
                ))

        if divergences:
            # Classify: is this a bug or expected non-determinism?
            for d in divergences:
                if d.field in spec.deterministic_fields():
                    # current_term, voted_for, commit_index must match exactly
                    report.add_error(d)
                elif d.field in spec.nondeterministic_fields():
                    # election_timeout can differ (random choice)
                    report.add_info(d)

    return report
```

**Handling non-determinism in conformance**: Some state differences are expected (random timer values, tie-breaking choices). The spec language marks which fields must match exactly vs. which can vary:

```
role Follower {
  state {
    current_term: int       @deterministic  // must match
    voted_for: node_id?     @deterministic  // must match
    log: Log                @deterministic  // must match
    election_timeout: float @nondeterministic  // can vary
  }
}
```

**Level 3: Statistical Conformance (performance drift)**

```python
def check_statistical_conformance(production_metrics, simulation_metrics):
    """
    Compare latency/throughput distributions between production and simulation.
    Uses Kolmogorov-Smirnov test for distribution comparison.
    """
    for metric_name in ["read_latency", "write_latency", "replication_lag"]:
        prod_dist = production_metrics[metric_name]
        sim_dist = simulation_metrics[metric_name]

        ks_stat, p_value = scipy.stats.ks_2samp(prod_dist, sim_dist)

        if p_value < 0.01:  # statistically significant difference
            # Is production WORSE than simulation?
            if prod_dist.percentile(99) > sim_dist.percentile(99) * 1.5:
                report.add_warning(PerformanceDrift(
                    metric=metric_name,
                    production_p99=prod_dist.percentile(99),
                    simulation_p99=sim_dist.percentile(99),
                    interpretation="Production is 50%+ slower than model predicts. "
                                   "Hardware degradation? Unexpected load pattern?"
                ))
            else:
                report.add_info(PerformanceImprovement(
                    metric=metric_name,
                    message="Production is faster than model. Update simulation model."
                ))
```

### Dual Execution: Injectable I/O In Practice

**The fundamental design**: All I/O goes through traits/interfaces. Production provides real implementations; simulation provides simulated ones.

```rust
// ─── The I/O Boundary ─────────────────────────────────
// Everything below this line is generated from the spec.
// Everything above is the implementation runtime.

trait RaftIO: Send + Sync {
    // Network
    async fn send_message(&self, to: NodeId, msg: RaftMessage) -> Result<()>;
    async fn receive_message(&self) -> Result<(NodeId, RaftMessage)>;

    // Time
    fn now(&self) -> Instant;
    async fn sleep_until(&self, deadline: Instant);
    fn random_duration(&self, min: Duration, max: Duration) -> Duration;

    // Persistent Storage
    async fn persist_state(&self, state: &PersistentState) -> Result<()>;
    async fn load_state(&self) -> Result<PersistentState>;
    async fn append_log(&self, entries: &[LogEntry]) -> Result<()>;
    async fn truncate_log(&self, from_index: u64) -> Result<()>;
}

// ─── Production Runtime ───────────────────────────────
struct ProductionIO {
    tcp_connections: HashMap<NodeId, TcpStream>,
    storage: RocksDb,
}

impl RaftIO for ProductionIO {
    async fn send_message(&self, to: NodeId, msg: RaftMessage) -> Result<()> {
        let conn = self.tcp_connections.get(&to)?;
        let bytes = msg.serialize();
        conn.write_all(&bytes).await?;
        Ok(())
    }
    // ... real implementations for everything
}

// ─── Simulation Runtime ───────────────────────────────
struct SimulationIO {
    bridge: Arc<SimulationBridge>,
    node_id: NodeId,
}

impl RaftIO for SimulationIO {
    async fn send_message(&self, to: NodeId, msg: RaftMessage) -> Result<()> {
        // Route through the simulation engine's Network entity
        // The simulation controls latency, loss, partitions
        self.bridge.schedule_message(self.node_id, to, msg).await
    }

    fn now(&self) -> Instant {
        // Returns simulation time, not wall clock
        self.bridge.sim_clock.now()
    }

    async fn sleep_until(&self, deadline: Instant) {
        // Yields to simulation scheduler (instant in sim time)
        self.bridge.yield_until(self.node_id, deadline).await
    }

    fn random_duration(&self, min: Duration, max: Duration) -> Duration {
        // Deterministic: uses seeded RNG controlled by simulation
        self.bridge.seeded_rng.gen_range(min..max)
    }
    // ...
}
```

**The key insight**: The Raft protocol logic (election, replication, commit advancement) is written ONCE, parameterized by `impl RaftIO`. The same code runs in:
- Unit tests with a mock IO
- Simulation with deterministic IO (fault injection, reproducible)
- Production with real IO

**What changes between simulation and production:**

| Concern | Simulation | Production |
|---------|-----------|------------|
| Network | Simulated latency + loss + partitions | Real TCP |
| Time | Simulated clock (jump to next event) | Real wall clock |
| Disk | In-memory with simulated fsync latency | Real RocksDB + fsync |
| Randomness | Seeded, deterministic | `/dev/urandom` |
| Failures | Injected by simulation controller | Real hardware failures |

**What does NOT change:**
- Protocol state machine logic
- Message serialization format
- Invariant checking code (runs in both modes)
- Log structure and operations

### Handling the "Escape Hatch" Gap

The generated code has well-defined extension points where custom logic plugs in:

```rust
// ─── Generated (correct by construction) ──────────────
// This code is mechanically derived from the spec.
// It handles: message dispatch, state transitions, timer management.

impl<IO: RaftIO> RaftNode<IO> {
    fn handle_append_entries(&mut self, from: NodeId, msg: AppendEntries)
        -> Vec<RaftMessage>
    {
        // Generated: term check, step-down logic
        if msg.term > self.current_term {
            self.current_term = msg.term;
            self.state = RaftState::Follower;
            self.voted_for = None;
        }

        // Generated: log consistency check
        if !self.log.check_consistency(msg.prev_log_index, msg.prev_log_term) {
            return vec![AppendEntriesResponse {
                term: self.current_term, success: false, ..
            }];
        }

        // Generated: append entries
        self.log.append_entries(msg.entries);

        // ─── ESCAPE HATCH: apply committed entries ────
        // This is where the user's state machine lives.
        // The spec provides a contract; the user provides the implementation.
        let newly_committed = self.log.advance_commit(msg.leader_commit);
        for entry in newly_committed {
            let result = self.state_machine.apply(entry.command);
            //                  ^^^^^^^^^^^^^^^^
            //  User-provided. Contract: must be deterministic.
            //  Checked by: property-based testing + runtime assertion.
        }

        vec![AppendEntriesResponse {
            term: self.current_term, success: true, ..
        }]
    }
}

// ─── Escape Hatch Interface ───────────────────────────
// User implements this. Contract derived from spec.
trait StateMachine: Send + Sync {
    /// Apply a command and return the result.
    ///
    /// CONTRACT (from spec):
    ///   - Must be deterministic: same command → same result
    ///   - Must be total: never panics for any valid command
    ///   - Idempotent when replayed (for crash recovery)
    ///
    /// VERIFIED BY:
    ///   - Property-based test: apply(cmd) twice → same state
    ///   - Simulation: run 10000 commands, compare results across replicas
    ///   - Runtime: periodic state hash comparison across replicas
    fn apply(&mut self, command: Command) -> Result;
}
```

---

## Deployment Model: Technical Deep Dive

### The Deployment State Space

For a rolling update of N nodes with per-node states:
```
NodeState = Running_V1 | Draining | Stopped | Starting_V2 | Running_V2 | Crashed | Recovering
```

**Without failures**: The rollout controller constrains transitions to a linear path. For batch_size=1:
```
Step 0:  [V1, V1, V1, V1, V1]       — initial
Step 1:  [Drain, V1, V1, V1, V1]    — draining node-0
Step 2:  [Stop, V1, V1, V1, V1]     — node-0 stopped
Step 3:  [Start2, V1, V1, V1, V1]   — node-0 starting v2
Step 4:  [V2, V1, V1, V1, V1]       — node-0 running v2
Step 5:  [V2, Drain, V1, V1, V1]    — draining node-1
...
Step 20: [V2, V2, V2, V2, V2]       — complete
```

**20 states — trivially model-checked.**

**With failures**: Each step can branch into failure states:
```
Step 4 branches:
  [V2, V1, V1, V1, V1]          — normal (node-0 healthy)
  [Crashed, V1, V1, V1, V1]     — node-0 crashed after upgrade
  [V2, V1, V1, Crashed, V1]     — node-3 crashed independently
  [V2, V1, V1, V1, V1] + partition({0},{1,2,3,4})  — network split
```

With single-failure injection at each step: ~20 steps × ~N possible failures × ~N partition configs = **~500 states for N=5**. Still tractable.

With multi-failure combinations: grows combinatorially but can be bounded (e.g., "at most 2 simultaneous failures") — standard technique in model checking.

### Mixed-Version Protocol Verification: The Subtle Bugs

The most dangerous deployment bugs occur when two versions interpret the same message differently. The spec language catches these.

**Example 1: Quorum change (UNSAFE)**

```
version v3 {
  compatible_with v2 {
    changed "quorum_size" {
      v2: majority(n)           // (n/2) + 1 = 3 for n=5
      v3: super_majority(n)     // (2n/3) + 1 = 4 for n=5
    }
  }
}
```

Model checker creates mixed cluster: `[v2, v2, v2, v3, v3]`

```
Scenario found by model checker:
  1. node-0 (v2) is leader, term=5
  2. Partition: {node-0, node-1, node-2} | {node-3, node-4}
  3. node-0's quorum = 3 (v2 rules) → has quorum → can commit
  4. node-3 (v3) starts election in partition
  5. node-3's quorum = 4 (v3 rules) → can't form quorum → OK so far
  6. Partition heals. node-3 sees node-0's commits.
  7. ALL GOOD in this case.

  BUT: if partition is {node-0, node-1} | {node-2, node-3, node-4}:
  1. node-0 (v2) quorum = 3, has only 2 → can't commit
  2. node-3 (v3) quorum = 4, has only 3 → can't commit
  3. DEADLOCK: neither side can make progress!

Result: LIVENESS VIOLATION detected during mixed-version window.
Recommendation: quorum change requires atomic switchover, not rolling update.
```

**Example 2: Message field semantics change (UNSAFE)**

```
version v3 {
  compatible_with v2 {
    changed "commit_index semantics" {
      v2: commit_index means "highest index known committed by leader"
      v3: commit_index means "highest index committed AND applied by leader"
    }
  }
}
```

This is a semantic change that can't be caught by structural compatibility. The spec language handles it:

```
// v3 spec declares the semantic difference
version v3 {
  compatible_with v2 {
    semantic_change "commit_index" {
      // v2 followers apply entries up to commit_index
      // v3 followers apply entries up to commit_index
      // BUT v3's commit_index is lower (only applied entries)
      // → v2 followers of v3 leader may commit MORE than intended

      check {
        // In mixed cluster, v2 follower applies entries the v3 leader
        // hasn't applied yet. Is this safe?
        forall follower in nodes where follower.version == v2:
          forall leader in nodes where leader.version == v3 && leader.role == Leader:
            follower.last_applied <= leader.commit_index_v2_equivalent
            // This check FAILS if v3's commit_index < v2's commit_index
            // for the same set of committed entries
      }
    }
  }
}
```

**Example 3: New optional field (SAFE)**

```
version v3 {
  compatible_with v2 {
    added_field "priority" in AppendEntries {
      type: int
      required: false
      default: 0
      // v2 nodes: ignore unknown fields (safe by protobuf convention)
      // v3 nodes: use priority if present, default to 0 if not
    }
  }
}

// Model checker verifies:
// 1. Protocol invariants hold when priority is always 0 (v2 behavior) ✓
// 2. Protocol invariants hold when priority varies (v3 behavior) ✓
// 3. Protocol invariants hold in mixed cluster ✓
// Result: SAFE to deploy via rolling update
```

### Deployment Automation Generation

The spec generates three kinds of deployment artifacts:

**1. Kubernetes manifests:**
```yaml
# Generated from deployment spec
apiVersion: apps/v1
kind: Deployment
metadata:
  name: raft-cluster
  labels:
    spec-version: "v3"
    spec-compatible-with: "v2"
    spec-verified: "true"
    spec-verification-hash: "sha256:abc123..."
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1      # = N - quorum = 5 - 3 = 2, conservative: 1
      maxSurge: 0
  template:
    spec:
      containers:
      - name: raft-node
        readinessProbe:       # Generated from spec's health_check
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 3
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /alive
            port: 8080
          periodSeconds: 10
```

**2. Admission webhook (enforces spec at deploy time):**
```python
# Generated admission controller
class SpecVerificationWebhook:
    def validate_deployment_update(self, old_deploy, new_deploy):
        old_version = old_deploy.labels["spec-version"]
        new_version = new_deploy.labels["spec-version"]

        # Check that compatibility was verified
        verification_hash = new_deploy.labels.get("spec-verification-hash")
        if not self.verification_store.is_verified(old_version, new_version, verification_hash):
            return Deny("Deployment blocked: version transition "
                       f"{old_version}→{new_version} has not been verified. "
                       "Run: spec-verify deploy --from {old_version} --to {new_version}")

        # Check that K8s rolling update params satisfy spec invariants
        max_unavailable = new_deploy.strategy.rollingUpdate.maxUnavailable
        quorum_required = self.spec.quorum_size(new_deploy.replicas)
        min_available = new_deploy.replicas - max_unavailable

        if min_available < quorum_required:
            return Deny(f"Rolling update allows {max_unavailable} unavailable, "
                       f"but quorum requires {quorum_required}/{new_deploy.replicas}. "
                       f"Max unavailable must be ≤ {new_deploy.replicas - quorum_required}")

        return Allow()
```

**3. Runbook (human-readable, generated from spec):**
```markdown
# Deployment Runbook: Raft v2 → v3

## Pre-deployment Checklist
- [ ] Version compatibility verified (spec-verify result: SAFE)
- [ ] Mixed-version simulation passed (1000 runs, 0 invariant violations)
- [ ] Rollback tested in simulation (recovery time: < 5s p99)

## Deployment Procedure
1. Upgrade node-0 (us-east-1a)
   - Drain connections (timeout: 10s)
   - Stop raft process
   - Upgrade binary to v3
   - Start process
   - Wait for health check (3 consecutive passes)
   - Verify: cluster has quorum (≥3 running nodes)
   - Wait 30s (stabilization period)

2. [Repeat for nodes 1-4]

## Rollback Triggers (automated)
- Error rate > 1% → automatic rollback
- P99 latency > 2× baseline → automatic rollback
- Any node fails health check 3× consecutively → automatic rollback

## Rollback Procedure
If triggered: all nodes revert to v2 simultaneously.
Expected rollback time: < 60s (verified in simulation).
```

---

## Open Questions

These are the hardest unsolved design problems. Getting these right determines whether the project succeeds.

### 1. How much of the protocol logic lives in Tier 1 vs. escape hatches?

If Tier 1 is too restricted (pure state machines, no loops over data structures), it can't express real protocols. If it's too expressive (arbitrary computation), it can't be model-checked.

**P language's answer**: State machines with bounded non-determinism. Practical but can't express unbounded data structures.

**Proposed answer**: Restrict Tier 1 to operations on **abstract data types** (Log, Set, Map) with formally specified semantics. The model checker reasons about abstract operations; the implementation provides concrete data structure implementations that satisfy the abstract spec (this is a standard refinement approach).

### 2. How do you handle non-determinism in the simulation?

Real distributed systems have non-determinism from: network reordering, OS scheduling, timer granularity, hardware behavior. The simulation must explore these non-deterministic choices.

**FoundationDB's answer**: Randomized exploration with many seeds (probabilistic coverage).

**TLA+'s answer**: Exhaustive exploration of all interleavings (formal but doesn't scale).

**Proposed answer**: Hybrid. Model-check the protocol logic (Tier 1) exhaustively for small instances. Simulate the full system (Tier 1 + 2 + 3) with randomized exploration for large instances. Use coverage metrics to measure how much of the state space has been explored.

### 3. What language should the spec be written in?

**Option A**: Custom language (like the examples above). Most control, best formal properties, highest adoption barrier.

**Option B**: Embedded DSL in Python. Lowest barrier, can leverage happysimulator immediately, but harder to enforce restrictions and extract to model checkers.

**Option C**: Embedded DSL in a typed language (Rust, TypeScript). Better type-level guarantees than Python, can use session type libraries (Rumpsteak for Rust).

**Recommended**: Start with Option B (Python DSL on happysimulator) to validate the concept quickly. Graduate to Option A or C once the design stabilizes. The examples in this document use Option A syntax for clarity, but the initial implementation could be Python classes and decorators.

### 4. How do you handle the "escape hatch" verification gap?

The generated code is correct by construction. But the escape hatches (custom logic filling in the skeleton) are not. How do you verify them?

**Proposed approach**: Layered verification for escape hatches:
1. **Contracts** (pre/post conditions) derived from the spec → checked via property-based testing
2. **Simulation oracle** → run the full implementation in simulation, compare outputs against the spec's simulation
3. **Runtime monitoring** → contracts become runtime assertions in production

This means escape hatches are **not proven** but they are:
- Specified (contracts say what they must do)
- Tested (property-based + simulation)
- Monitored (runtime assertions)

For LLM-generated escape hatch code, this is strong enough: the LLM generates code, the contract tests reject incorrect implementations, the simulation validates behavior.

---

## Recommendation: If You Build This, Here's the Path

### Phase 0: Validate (weeks, not months)
Write a Raft spec in the proposed syntax (on paper). Manually translate it to:
- TLA+ → verify it model-checks correctly against known Raft TLA+ spec
- Python/happysimulator → verify it simulates correctly with fault injection
- Show that the SAME spec feeds both tools

This validates the core thesis: one spec, multiple verification backends.

### Phase 1: Python DSL Prototype
Build the spec language as a Python embedded DSL:
```python
@role
class Follower:
    current_term: int = 0
    voted_for: Optional[NodeId] = None

    @on("RequestVote")
    def handle_vote(self, msg, sender):
        ...

@invariant
def one_leader_per_term(nodes):
    leaders = [n for n in nodes if n.role == "Leader"]
    return len(set(l.current_term for l in leaders)) == len(leaders)

@deployment("rolling")
class RaftRollingUpdate:
    batch_size: int = 1
    ...
```

Compile this to happysimulator entities. Run simulations. Check invariants. This gives you **Tier 1 + Tier 2** with simulation-based checking.

### Phase 2: Model Checker Extraction
Add a TLA+ backend: translate the `@role` state machines and `@invariant` properties to TLA+. Run TLC. Compare results with simulation — they should agree on safety violations.

### Phase 3: Session Types
Add protocol annotations. Generate session-typed interfaces. This is the hardest engineering step but highest-value for the "LLM generates implementation" story.

### Phase 4: Deployment Model
Add `@deployment` specs. Model-check deployment invariants. Generate K8s manifests. Simulate rolling updates.

### Phase 5: Trace Conformance
Build the instrumentation SDK. Implement trace replay. Deploy runtime monitors.

Each phase produces standalone value. You don't need all five to have something useful.

---

---

# Part 4: Concrete Prototype Sketch

---

## Goal

Demonstrate the core thesis with the minimum amount of code: **one spec → simulation backend + TLA+ extraction**. Use a simplified Raft election (no log replication) as the example protocol.

## File Structure

```
specverify/                          # The prototype package
├── __init__.py                      # Public API
├── dsl/
│   ├── __init__.py
│   ├── decorators.py                # @role, @message, @handler, @invariant
│   ├── types.py                     # NodeId, Timer, state field descriptors
│   └── spec.py                      # SystemSpec: collects all declarations
├── backends/
│   ├── __init__.py
│   ├── simulation.py                # Compile spec → happysimulator entities
│   └── tla.py                       # Compile spec → TLA+ module
├── examples/
│   └── leader_election/
│       ├── spec.py                  # The election spec in the DSL
│       ├── test_simulation.py       # Run simulation, check invariants
│       └── test_tla_extraction.py   # Extract TLA+, model check
└── tests/
    ├── test_dsl.py
    ├── test_simulation_backend.py
    └── test_tla_backend.py
```

## The DSL: What Users Write

```python
# specverify/examples/leader_election/spec.py

from specverify import (
    message, role, handler, invariant, timer,
    SystemSpec, NodeId, broadcast, reply_to,
)

# ─── Messages ──────────────────────────────────────────

@message
class RequestVote:
    term: int
    candidate_id: NodeId
    # (simplified: no log index/term for this prototype)

@message
class VoteResponse:
    term: int
    vote_granted: bool

# ─── Roles ─────────────────────────────────────────────

@role
class Follower:
    """A node that follows a leader."""

    # State fields (become Entity instance variables)
    current_term: int = 0
    voted_for: NodeId | None = None

    # Timers
    election_timeout = timer(min_s=1.5, max_s=3.0)

    @handler(RequestVote)
    def on_request_vote(self, msg: RequestVote, sender: NodeId):
        """Handle a vote request from a candidate."""
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.voted_for = None

        grant = (
            msg.term >= self.current_term
            and (self.voted_for is None or self.voted_for == sender)
        )

        if grant:
            self.voted_for = sender
            self.election_timeout.reset()  # heard from a candidate

        return reply_to(sender, VoteResponse(
            term=self.current_term,
            vote_granted=grant,
        ))

    @handler(election_timeout.expired)
    def on_election_timeout(self):
        """No heartbeat received; become candidate."""
        return self.transition_to(Candidate)


@role
class Candidate:
    """A node seeking election."""

    current_term: int     # inherited from previous role
    voted_for: NodeId | None
    votes_received: set[NodeId] = set()

    election_timeout = timer(min_s=1.5, max_s=3.0)

    @handler("enter")
    def on_enter(self):
        """Transition logic: increment term, vote for self, request votes."""
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.election_timeout.reset()

        return broadcast(RequestVote(
            term=self.current_term,
            candidate_id=self.node_id,
        ))

    @handler(VoteResponse)
    def on_vote_response(self, msg: VoteResponse, sender: NodeId):
        if msg.term > self.current_term:
            self.current_term = msg.term
            return self.transition_to(Follower)

        if msg.vote_granted:
            self.votes_received.add(sender)

        if len(self.votes_received) >= self.quorum_size:
            return self.transition_to(Leader)

        return None

    @handler(election_timeout.expired)
    def on_election_timeout(self):
        """Election failed; restart."""
        return self.transition_to(Candidate)  # new election

    @handler(RequestVote)
    def on_request_vote(self, msg: RequestVote, sender: NodeId):
        """Step down if we see a higher term."""
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.voted_for = None
            return self.transition_to(Follower)
        return reply_to(sender, VoteResponse(
            term=self.current_term, vote_granted=False
        ))


@role
class Leader:
    """Elected leader."""

    current_term: int
    voted_for: NodeId | None

    heartbeat_timer = timer(interval_s=0.5)

    @handler("enter")
    def on_enter(self):
        self.heartbeat_timer.start()
        return None  # (would send initial heartbeat in full impl)

    @handler(heartbeat_timer.tick)
    def on_heartbeat(self):
        # (Simplified: no AppendEntries, just maintain leadership)
        return None

    @handler(RequestVote)
    def on_request_vote(self, msg: RequestVote, sender: NodeId):
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.voted_for = None
            return self.transition_to(Follower)
        return reply_to(sender, VoteResponse(
            term=self.current_term, vote_granted=False
        ))


# ─── Invariants ────────────────────────────────────────

@invariant
def at_most_one_leader_per_term(nodes):
    """Safety: no two nodes are Leader in the same term."""
    leaders = [(n.node_id, n.current_term) for n in nodes if n.current_role == Leader]
    terms = [term for _, term in leaders]
    return len(terms) == len(set(terms))  # all terms unique


@invariant
def votes_consistent(nodes):
    """If a node voted for candidate X, then X is a Candidate or Leader."""
    for n in nodes:
        if n.voted_for is not None and n.voted_for != n.node_id:
            voter_term = n.current_term
            candidate = next((m for m in nodes if m.node_id == n.voted_for), None)
            if candidate:
                assert candidate.current_term >= voter_term


# ─── System Spec ───────────────────────────────────────

election_spec = SystemSpec(
    name="LeaderElection",
    roles=[Follower, Candidate, Leader],
    initial_role=Follower,
    invariants=[at_most_one_leader_per_term, votes_consistent],
    messages=[RequestVote, VoteResponse],
)
```

## Backend 1: Compilation to happysimulator

The `simulation.py` backend translates the spec into happysimulator `Entity` subclasses.

```python
# specverify/backends/simulation.py

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.components.network import Network, NetworkLink
import random

class SpecEntity(Entity):
    """An Entity generated from a spec role.

    Each SpecEntity instance represents one node.
    It holds:
      - current_role: which @role class is active
      - state: the union of all role state fields
      - handlers: message → handler function dispatch table
      - network: for sending messages to peers
      - peers: list of peer SpecEntity instances
    """

    def __init__(self, name, spec, network, peers=None):
        super().__init__(name)
        self.spec = spec
        self._network = network
        self._peers = peers or []
        self.node_id = name

        # Initialize with the spec's initial role
        self.current_role = spec.initial_role
        self._state = {}
        self._timers = {}
        self._init_role_state(spec.initial_role)

    def _init_role_state(self, role_cls):
        """Initialize state fields from a @role class."""
        for field_name, default in role_cls.__spec_fields__.items():
            if field_name not in self._state:  # preserve across transitions
                self._state[field_name] = default

        # Set up timers
        for timer_name, timer_config in role_cls.__spec_timers__.items():
            self._schedule_timer(timer_name, timer_config)

    @property
    def quorum_size(self):
        return (len(self._peers) + 1) // 2 + 1

    def __getattr__(self, name):
        """Allow handlers to access state fields directly."""
        if name.startswith('_') or name in ('spec', 'node_id', 'current_role', 'name'):
            raise AttributeError(name)
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"No state field '{name}'")

    def __setattr__(self, name, value):
        if hasattr(self, '_state') and name in self._state:
            self._state[name] = value
        else:
            super().__setattr__(name, value)

    def handle_event(self, event: Event):
        et = event.event_type

        # Timer events
        if et.startswith("_timer:"):
            timer_name = et.split(":", 1)[1]
            handler = self.current_role.__spec_handlers__.get(f"timer:{timer_name}")
            if handler:
                result = handler(self)
                return self._process_result(result, event)
            return None

        # Role transition events
        if et == "_role_transition":
            new_role = event.context["metadata"]["new_role"]
            self.current_role = new_role
            self._init_role_state(new_role)
            enter_handler = new_role.__spec_handlers__.get("enter")
            if enter_handler:
                result = enter_handler(self)
                return self._process_result(result, event)
            return None

        # Message events
        msg_type = et  # event_type == message class name
        handler = self.current_role.__spec_handlers__.get(msg_type)
        if handler:
            msg_data = event.context["metadata"].get("payload")
            sender = event.context["metadata"].get("source")
            result = handler(self, msg_data, sender)
            return self._process_result(result, event)

        return None

    def _process_result(self, result, event):
        """Convert DSL results (reply_to, broadcast, transition_to) to Events."""
        if result is None:
            return []

        events = []
        if isinstance(result, list):
            for r in result:
                events.extend(self._process_single_result(r, event))
        else:
            events.extend(self._process_single_result(result, event))
        return events

    def _process_single_result(self, result, event):
        """Convert a single result to Event(s)."""
        events = []
        if isinstance(result, ReplyAction):
            target = self._find_peer(result.to)
            events.append(self._network.send(
                source=self, destination=target,
                event_type=type(result.message).__name__,
                payload=result.message.__dict__,
            ))
        elif isinstance(result, BroadcastAction):
            for peer in self._peers:
                events.append(self._network.send(
                    source=self, destination=peer,
                    event_type=type(result.message).__name__,
                    payload=result.message.__dict__,
                ))
        elif isinstance(result, TransitionAction):
            events.append(Event(
                time=self.now,
                event_type="_role_transition",
                target=self,
                context={"metadata": {"new_role": result.role}},
            ))
        return events

    def _schedule_timer(self, name, config):
        """Schedule a timer event."""
        if config.get("interval_s"):
            delay = config["interval_s"]
        else:
            delay = random.uniform(config["min_s"], config["max_s"])

        evt = Event(
            time=self.now + delay,
            event_type=f"_timer:{name}",
            target=self,
            daemon=True,
        )
        self._timers[name] = evt
        return evt

    def _find_peer(self, node_id):
        for p in self._peers:
            if p.node_id == node_id:
                return p
        raise ValueError(f"Unknown peer: {node_id}")


def compile_to_simulation(spec, n_nodes=3, duration=60.0, seed=42):
    """Compile a SystemSpec into a runnable happysimulator Simulation."""
    random.seed(seed)

    # Create network
    network = Network(name="cluster")

    # Create nodes
    nodes = []
    for i in range(n_nodes):
        node = SpecEntity(f"node-{i}", spec, network)
        nodes.append(node)

    # Wire peers
    for node in nodes:
        node._peers = [n for n in nodes if n is not node]

    # Add network links
    from happysimulator.distributions import ConstantLatency
    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            link = NetworkLink(
                name=f"link-{a.name}-{b.name}",
                latency=ConstantLatency(0.001),  # 1ms
            )
            network.add_bidirectional_link(a, b, link)

    # Create simulation
    all_entities = [network] + nodes
    sim = Simulation(
        duration=duration,
        entities=all_entities,
    )

    # Schedule initial timer events for all nodes
    for node in nodes:
        for timer_name, config in spec.initial_role.__spec_timers__.items():
            evt = node._schedule_timer(timer_name, config)
            sim.schedule(evt)

    # Install invariant checking (check after every non-daemon event)
    for inv_fn in spec.invariants:
        def make_checker(fn):
            def check(event):
                if not event.daemon:
                    try:
                        result = fn(nodes)
                        if result is False:
                            raise AssertionError(
                                f"Invariant '{fn.__name__}' violated "
                                f"at t={event.time}"
                            )
                    except AssertionError:
                        raise
                    except Exception:
                        pass  # invariant raised internally
            return check
        sim.control.on_event(make_checker(inv_fn))

    return sim, nodes
```

**Usage:**
```python
# specverify/examples/leader_election/test_simulation.py

from leader_election.spec import election_spec
from specverify.backends.simulation import compile_to_simulation

def test_leader_elected():
    """A leader should be elected within the first few seconds."""
    sim, nodes = compile_to_simulation(election_spec, n_nodes=3, duration=30.0)
    sim.run()

    leaders = [n for n in nodes if n.current_role.__name__ == "Leader"]
    assert len(leaders) >= 1, "No leader was elected"

def test_safety_under_partition():
    """Safety: at most one leader per term, even with network partitions."""
    sim, nodes = compile_to_simulation(election_spec, n_nodes=5, duration=60.0)

    # Inject partition at t=10
    network = sim._entities[0]
    sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(10)))
    sim.run()

    partition = network.partition([nodes[0]], nodes[1:])

    sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(40)))
    sim.run()

    partition.heal()
    sim.control.resume()
    summary = sim.run()

    # Invariants were checked continuously via on_event hook
    # If we get here, no invariant was violated
    assert summary.total_events_processed > 0
```

## Backend 2: TLA+ Extraction

The `tla.py` backend translates the spec into a TLA+ module for model checking.

```python
# specverify/backends/tla.py

def compile_to_tla(spec, max_nodes=3, max_term=4):
    """Translate a SystemSpec into a TLA+ module string."""

    roles = {r.__name__: r for r in spec.roles}
    messages = {m.__name__: m for m in spec.messages}

    tla = []
    tla.append(f"---- MODULE {spec.name} ----")
    tla.append(f"EXTENDS Naturals, FiniteSets, Sequences")
    tla.append("")

    # Constants
    tla.append(f"CONSTANTS")
    tla.append(f"  Nodes,        \\* Set of node identifiers")
    tla.append(f"  MaxTerm       \\* Bound for model checking")
    tla.append("")

    # Variables (derived from role state fields)
    all_fields = set()
    for role in spec.roles:
        all_fields.update(role.__spec_fields__.keys())

    tla.append("VARIABLES")
    tla.append(f"  currentRole,  \\* function: node -> role")
    for field in sorted(all_fields):
        tla.append(f"  {field},")
    tla.append(f"  messages       \\* set of in-flight messages")
    tla.append("")

    # State space
    tla.append(f'Roles == {{"Follower", "Candidate", "Leader"}}')
    tla.append("")

    # Init
    tla.append("Init ==")
    tla.append(f'  /\\ currentRole = [n \\in Nodes |-> "Follower"]')
    for field, default in spec.initial_role.__spec_fields__.items():
        if isinstance(default, (int, float)):
            tla.append(f"  /\\ {field} = [n \\in Nodes |-> {default}]")
        elif default is None:
            tla.append(f'  /\\ {field} = [n \\in Nodes |-> "None"]')
        elif isinstance(default, set):
            tla.append(f"  /\\ {field} = [n \\in Nodes |-> {{}}]")
    tla.append("  /\\ messages = {}")
    tla.append("")

    # Actions (derived from @handler methods)
    for role in spec.roles:
        for handler_name, handler_fn in role.__spec_handlers__.items():
            action_name = f"{role.__name__}_{handler_name}"
            tla_action = _translate_handler_to_tla(
                action_name, role, handler_name, handler_fn, spec
            )
            tla.append(tla_action)
            tla.append("")

    # Next relation (disjunction of all actions)
    tla.append("Next ==")
    actions = []
    for role in spec.roles:
        for handler_name in role.__spec_handlers__:
            actions.append(f"  \\/ \\E n \\in Nodes: {role.__name__}_{handler_name}(n)")
    tla.append("\n".join(actions))
    tla.append("")

    # Invariants (translated from @invariant functions)
    for inv_fn in spec.invariants:
        tla_inv = _translate_invariant_to_tla(inv_fn)
        tla.append(tla_inv)
        tla.append("")

    # Spec
    tla.append("Spec == Init /\\ [][Next]_vars")
    tla.append("")
    tla.append("====")

    return "\n".join(tla)


def _translate_handler_to_tla(action_name, role, handler_name, handler_fn, spec):
    """
    Translate a Python handler function to a TLA+ action.

    This is the HARD part. The prototype handles a restricted subset:
    - Simple if/elif conditions on message fields and state
    - Direct state assignments
    - reply_to / broadcast return values
    - transition_to calls

    Complex logic (loops, function calls) falls back to manual annotation.
    """
    # For the prototype: use AST analysis of the handler function
    # to extract the guarded-command structure.
    #
    # Full implementation would use a restricted Python subset
    # that's guaranteed to be translatable.

    import ast
    import inspect
    source = inspect.getsource(handler_fn)
    tree = ast.parse(source)

    # ... AST → TLA+ translation logic ...
    # (This is where the real engineering effort goes)

    # For the prototype, emit a template with manual sections:
    lines = [f"{action_name}(n) =="]
    lines.append(f'  /\\ currentRole[n] = "{role.__name__}"')

    if handler_name in [m.__name__ for m in spec.messages]:
        # Message handler: requires message in messages set
        lines.append(f'  /\\ \\E m \\in messages:')
        lines.append(f'       /\\ m.type = "{handler_name}"')
        lines.append(f'       /\\ m.to = n')
        lines.append(f'       \\* ... translated guard conditions ...')
        lines.append(f'       \\* ... translated state updates ...')
        lines.append(f'       \\* ... translated message sends ...')

    return "\n".join(lines)


def _translate_invariant_to_tla(inv_fn):
    """Translate a Python invariant function to a TLA+ invariant."""
    # For the prototype: pattern-match common invariant shapes

    # Example: at_most_one_leader_per_term
    # Python: len(terms) == len(set(terms))
    # TLA+:  \A n1, n2 \in Nodes:
    #           (currentRole[n1] = "Leader" /\ currentRole[n2] = "Leader")
    #           => (n1 = n2 \/ currentTerm[n1] /= currentTerm[n2])

    name = inv_fn.__name__
    return (
        f"{name} ==\n"
        f"  \\* AUTO-TRANSLATED from Python @invariant\n"
        f"  \\* Manual verification recommended\n"
        f"  \\A n1, n2 \\in Nodes:\n"
        f"    ... \\* TODO: auto-translate from AST"
    )
```

## What the Prototype Demonstrates

With this prototype, you can:

1. **Write a protocol spec once** (the `election_spec` in Python)
2. **Run it as a simulation** (`compile_to_simulation` → happysimulator)
3. **Extract to TLA+** (`compile_to_tla` → TLA+ module)
4. **Check invariants in both backends** (simulation hook + TLC model checker)
5. **Inject faults in simulation** (network partitions, node crashes)
6. **Compare results**: Do both backends agree on invariant violations?

## What the Prototype Does NOT Do (Yet)

- No session types (Phase 3 of the roadmap)
- No code generation to Rust/Go (Phase 3)
- No deployment model (Phase 4)
- No trace conformance (Phase 5)
- TLA+ extraction is semi-automatic (AST translation is the hardest part)
- No model checker integration (manual TLC run)

## Effort Estimate for Prototype

| Component | Effort | Notes |
|-----------|--------|-------|
| DSL decorators (`@role`, `@message`, etc.) | 2-3 days | Metaclass/descriptor magic |
| Simulation backend | 3-5 days | Map DSL constructs to happysimulator entities |
| TLA+ extraction (basic) | 5-7 days | AST analysis + TLA+ codegen. Hardest part. |
| Example: leader election | 1 day | Write the spec + tests |
| Example: with fault injection | 1 day | Partition tests |
| **Total** | **~2-3 weeks** | For a working demo |

## What Success Looks Like

The prototype is successful if:

1. You can write a leader election spec in ~50 lines of Python
2. `compile_to_simulation(spec).run()` produces correct behavior with fault injection
3. `compile_to_tla(spec)` produces a TLA+ module that TLC can check
4. An intentional bug in the spec (e.g., wrong quorum calculation) is caught by BOTH backends
5. A timing-related issue (e.g., election timeout too short) is caught by simulation but NOT by TLA+ (demonstrating complementary value)

Point 5 is the key demo: **the simulation catches bugs that TLA+ can't**, validating the need for both.

---

## Sources

**Deterministic Simulation Testing:**
- [FoundationDB Simulation and Testing](https://apple.github.io/foundationdb/testing.html)
- [DST Primer for Unit Test Maxxers](https://www.amplifypartners.com/blog-posts/a-dst-primer-for-unit-test-maxxers)
- [WarpStream: DST for Entire SaaS](https://www.warpstream.com/blog/deterministic-simulation-testing-for-our-entire-saas)

**Formal Specification Languages:**
- [P Language (Microsoft)](https://github.com/p-org/P) — state machine spec + code gen, used in AWS S3
- [P: Asynchrony, Fault-tolerance, Uncertainty](https://www.microsoft.com/en-us/research/blog/p-programming-language-asynchrony/)

**Verified Distributed Systems:**
- [IronFleet: Proving Practical Distributed Systems Correct](https://www.andrew.cmu.edu/user/bparno/papers/ironfleet.pdf)
- [Verdi: Verified Distributed Systems in Coq](https://homes.cs.washington.edu/~ztatlock/pubs/verdi-wilcox-pldi15.pdf)
- [Smart Casual Verification (2025)](https://decentralizedthoughts.github.io/2025-05-23-smart-casual-verification/)

**Session Types:**
- [Multiparty Session Types with Crash-Stop Failures (ECOOP 2023)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECOOP.2023.1)
- [Hybrid Multiparty Session Types](https://dl.acm.org/doi/10.1145/3586031) — Scribble + Teatrino toolchain
- [Programming Language Implementations with MPST](https://link.springer.com/chapter/10.1007/978-3-031-51060-1_6) — includes Rumpsteak for Rust

**Refinement Types:**
- [Verifying Replicated Data Types with Liquid Haskell](https://dl.acm.org/doi/10.1145/3428284)
- [Adventures in Reliable Distributed Systems with Liquid Haskell (FLOPS 2022)](https://decomposition.al/blog/2022/07/20/my-flops-2022-keynote-talk-adventures-in-building-reliable-distributed-systems-with-liquid-haskell/)

**Digital Twins & Runtime Verification:**
- [Formal Verification of Digital Twins with TLA](https://arxiv.org/html/2411.18798v1)
- [Contract-Based Verification of Digital Twins](https://link.springer.com/chapter/10.1007/978-3-032-00828-2_19)
- [Refinement-based Runtime Validation](https://ar5iv.labs.arxiv.org/html/1703.05317)
