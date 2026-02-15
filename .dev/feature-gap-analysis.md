# Feature Gap Analysis: happysimulator vs. Competitors

## Context

happysimulator is a comprehensive DES library with ~100+ components spanning distributed systems, infrastructure, industrial, behavioral, and streaming simulation. This analysis compares against SimPy, MATLAB SimEvents, Salabim, AnyLogic, and Ciw to identify genuine gaps and propose new features.

**What's already strong:** Entity/actor model, generator coroutines with SimFuture, nanosecond time, 10+ queue policies, full distributed systems stack (Raft/Paxos/CRDTs/replication), infrastructure primitives (Disk/CPU/GC/TCP), industrial components (20 examples), behavioral modeling (agents/populations/social graphs), visual debugger, AI integration layer, analysis tools, 50+ examples.

---

## Tier 1: High Impact, Broad Benefit (Implement First)

### 1. Multiple Replications with Statistical Analysis
**Gap source:** General DES practice (all commercial tools have this)
**What:** Run N independent replications with different seeds, compute confidence intervals, detect convergence automatically.
**Why:** Without this, results are unreliable -- could be noise from a single seed. Every serious simulation study requires it. Currently users must write custom orchestration.
**Size:** Medium | **Fits:** Extends existing `ai/result.py` (`SweepResult`, `SimulationResult`)
**Implementation:**
- New package `happysimulator/experiment/`
- `ReplicationRunner(scenario_fn, n_replications, seed_strategy)` orchestrates N `Simulation.run()` calls
- `ReplicationResult` aggregates N `SimulationResult` objects with CI computation
- Statistical helpers: confidence intervals, convergence detection (relative half-width), Welch's t-test
- Integration with existing `SweepResult` for parameter sweep + replications

### 2. Warm-up Period Detection
**Gap source:** General DES practice, Ciw
**What:** Auto-detect when initial transient bias dissipates. Methods: MSER-5, Welch's graphical, batch means.
**Why:** Nearly all results are contaminated by initial conditions unless warm-up is identified and discarded. Users currently guess `between(start, end)` parameters.
**Size:** Small-Medium | **Fits:** Extends `analysis/phases.py` pattern (windowed stats over `Data`)
**Implementation:**
- New module `happysimulator/analysis/warmup.py`
- `detect_warmup(data, method="mser5") -> WarmupResult(truncation_point_s, method, confidence)`
- Pairs with replications for rigorous steady-state analysis

### 3. Scenario Management
**Gap source:** AnyLogic experiment framework, general DES practice
**What:** Define named parameter sets (scenarios), run them systematically, compare results.
**Why:** Current `SweepResult` only sweeps a single parameter. Real studies need multi-parameter configs ("baseline" vs. "with_caching" vs. "high_load").
**Size:** Medium | **Fits:** Extends `ai/result.py` pattern
**Implementation:**
- `Scenario(name, parameters, seed, description)` dataclass
- `ExperimentRunner(scenario_factory_fn, scenarios, n_replications=1)`
- `ExperimentResult` with cross-scenario comparison matrix and `to_prompt_context()`

### 4. Container / Tank Resource
**Gap source:** SimPy (`Container`)
**What:** Models homogeneous bulk material (fuel, water, power, money) as a continuous level with `put(amount)` / `get(amount)`. Unlike `Resource` (discrete slots), this has a `level` and `capacity` ceiling. Puts block when full; gets block when insufficient.
**Why:** Most frequently needed SimPy feature with no equivalent. Models: energy budgets, buffer pools, material flow, cash registers, water tanks.
**Size:** Medium | **Fits:** Mirrors existing `Resource` pattern exactly (Entity + SimFuture waiters)
**Implementation:**
- New file `happysimulator/components/container.py`
- `Container(name, capacity, initial_level=0)` with `put(amount) -> SimFuture`, `get(amount) -> SimFuture`
- `try_put(amount) -> bool`, `try_get(amount) -> float | None` (non-blocking)
- `level`, `capacity`, `utilization`, `stats` properties
- Two waiter queues (putters + getters) with wake logic from `resource.py`

### 5. Process Interruption
**Gap source:** SimPy (process interrupts)
**What:** Allow one entity to interrupt another's running generator, causing an `Interrupt` exception that can be caught or propagated.
**Why:** Essential for timeouts, preemption, cancellation, crash-during-processing. Currently generators can't be interrupted once waiting. The `any_of` timeout race requires anticipating every interruption at every yield point.
**Size:** Large | **Touches:** Core engine (`event.py` ProcessContinuation, `sim_future.py`)
**Implementation:**
- New `Process` handle class wrapping a generator, exposing `interrupt(cause)`
- `SimFuture.fail(exception)` method (resumes parked generators with `gen.throw()`)
- `ProcessContinuation` enhancements for throw semantics
- `Entity.handle_event()` returns a `Process` handle when it produces a generator
- `Interrupt` exception class with `cause` attribute

---

## Tier 2: Strong Value, Targeted Users (Implement Second)

### 6. Store / FilterStore
**Gap source:** SimPy (`Store`, `FilterStore`, `PriorityStore`)
**What:** Typed object buffer where entities `put(item)` and `get(filter_fn)` concrete Python objects with optional predicate-based retrieval.
**Why:** Models mailboxes, heterogeneous work buffers, object pools with typed selection. Current Queue stores Events, not arbitrary objects.
**Size:** Medium | **Fits:** Same Entity + SimFuture waiter pattern as Resource/Container

### 7. State-Dependent Arrivals & Routing
**Gap source:** Ciw, general DES practice
**What:** Arrival rates and routing that dynamically change based on current system state (queue lengths, utilization, etc.).
**Why:** Static rates are unrealistic. Real systems have backpressure, load-shedding, crowd-avoidance.
**Size:** Small-Medium | **Fits:** New `StateDependentProfile` implementing existing `Profile` protocol
**Implementation:**
- `Source.state_dependent(rate_fn=lambda state: 10 if state["depth"] < 50 else 2, state_fn=...)`
- Optional: `ProbabilisticRouter(targets, weight_fn)` for dynamic routing weights

### 8. Processor Sharing
**Gap source:** Ciw, queuing theory (one of 4 fundamental disciplines: FCFS/LCFS/PS/IS)
**What:** Server capacity shared equally among all present customers. Effective service rate decreases as more arrive.
**Why:** Correct model for web servers, CPU time-sharing, bandwidth sharing. The only fundamental service discipline happysimulator lacks.
**Size:** Medium-Large | **Fits:** New component, uses Event cancellation for rescheduling

### 9. Phase-Type Distributions (Erlang, Hyperexponential, Coxian)
**Gap source:** Ciw, queuing theory
**What:** Mathematically rich distribution family. Erlang = sum of exponentials (lower variance). Hyperexponential = mixture (higher variance). Coxian = general.
**Why:** Current distribution library lacks these standard OR/queuing theory distributions. Also add LogNormal and Gamma while at it.
**Size:** Small | **Fits:** New `LatencyDistribution` subclasses following `exponential.py` pattern

### 10. Jockeying
**Gap source:** Ciw
**What:** Customers switch between parallel queues to find shorter ones.
**Why:** Realistic multi-server behavior for bank/grocery/airport models. Without it, multi-queue models overstate variance.
**Size:** Small-Medium | **Fits:** New coordinator Entity monitoring existing `QueuedResource` depths

### 11. Deadlock Detection
**Gap source:** Ciw
**What:** Auto-detect when simulation is deadlocked (all entities waiting on each other in a cycle).
**Why:** Deadlocks are subtle bugs in models with multiple resources/futures. Currently a deadlocked simulation just hangs silently.
**Size:** Medium | **Fits:** Extends `Simulation._run_loop()` -- build wait-for graph from parked SimFutures, detect cycles

---

## Tier 3: Niche Value, Larger Scope (Implement If Demanded)

### 12. Real-Time Simulation
**Gap source:** SimPy (`RealtimeEnvironment`)
**What:** Synchronize sim-time to wall-clock (1 sim-second = 1 real-second). Speed factor supported.
**Why:** Hardware-in-the-loop, live demos, training simulations.
**Size:** Small-Medium | Add `real_time=True` flag to `Simulation`, sleep between events

### 13. State Objects (Observable State Variables)
**Gap source:** Salabim (`State`)
**What:** `StateVar(value)` entity. Entities `yield state.wait_for(predicate)`. On `set()`, matching waiters wake.
**Why:** More general than one-shot SimFuture for coordination. Persistent, multi-watcher, multi-condition.
**Size:** Small-Medium | Built on SimFuture internally

### 14. Simulation Checkpointing
**Gap source:** General DES practice
**What:** Save/restore simulation state for resume, branching, crash recovery.
**Why:** Long-running simulations, what-if branching from a specific time point.
**Size:** Large | Requires entity `get_state()/set_state()` protocol, no in-flight generators

### 15. Continuous-Discrete Hybrid (System Dynamics)
**Gap source:** MATLAB SimEvents + Simulink, AnyLogic
**What:** Continuous-time dynamics (ODEs, stock-and-flow) combined with discrete events. Models fluid levels, temperature, battery charge.
**Why:** Biggest paradigm gap vs. commercial tools. Many real systems have both continuous and discrete dynamics.
**Size:** Large | New `ContinuousVariable` Entity with periodic integration-step events, threshold-crossing events

---

## Tier 4: Deprioritize

| Feature | Reason |
|---------|--------|
| **2D Animation** | Enormous effort; existing visual debugger covers most needs. Lightweight alternative: add optional `position=(x,y)` to Entity for graph view layout |
| **Video Recording** | Depends on animation; screen-record the browser debugger instead |
| **GIS/Spatial** | Domain-specific (logistics/transport); users can use coordinates in event context |
| **Optimization Framework** | Better served by integrating with Optuna/scipy.optimize via a thin adapter |
| **Entity Multicasting** | Already mostly exists via `SplitMerge` and `Topic`. Add `Event.broadcast()` helper if needed |

---

## Recommended Build Order

```
Phase 1 - Experiment Infrastructure + Core Resources:
  #1  Multiple Replications with Statistical Analysis
  #2  Warm-up Period Detection
  #4  Container/Tank Resource
  #9  Phase-Type Distributions (Erlang, Hyperexponential, LogNormal, Gamma)

Phase 2 - Core Gaps + Experiment Extension:
  #3  Scenario Management (extends replications)
  #5  Process Interruption
  #6  Store/FilterStore
  #7  State-Dependent Arrivals/Routing

Phase 3 - Advanced Components:
  #8  Processor Sharing
  #10 Jockeying
  #11 Deadlock Detection
  #13 State Objects

Phase 4 - Domain Extensions (if demanded):
  #15 Continuous-Discrete Hybrid
  #12 Real-Time Simulation
  #14 Checkpointing
```

Features within each phase have no inter-dependencies and can be developed in parallel.

---

## Key Files to Reference During Implementation

| Feature | Pattern to follow |
|---------|------------------|
| Container, Store | `happysimulator/components/resource.py` (Entity + SimFuture waiter queues) |
| Process Interruption | `happysimulator/core/event.py` (ProcessContinuation), `happysimulator/core/sim_future.py` |
| Replications, Scenarios | `happysimulator/ai/result.py` (SimulationResult, SweepResult, SimulationComparison) |
| Warm-up Detection | `happysimulator/analysis/phases.py` (windowed statistical analysis over Data) |
| Phase-Type Distributions | `happysimulator/distributions/exponential.py` (LatencyDistribution subclass) |
| State-Dependent Arrivals | `happysimulator/load/source.py`, `happysimulator/load/profile.py` (Profile protocol) |
| Deadlock Detection | `happysimulator/core/simulation.py` (_run_loop), `happysimulator/core/sim_future.py` |

## Verification

For each implemented feature:
1. Unit tests in `tests/unit/` following existing patterns
2. Integration test in `tests/integration/` composing with other components
3. Dedicated example in `examples/` demonstrating the feature's key insight
4. Run `pytest -q` to ensure no regressions
5. Update `CLAUDE.md` with new component documentation
