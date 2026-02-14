# Plan: Industrial Simulation Components & Examples

## Context

The happy-simulator library is a discrete-event simulator designed for distributed systems. The user wants to expand it with classic operations research / industrial engineering simulations (bank tellers, manufacturing lines, etc.). This requires new reusable components for behaviors common in industrial settings but not covered by the existing distributed-systems-focused component library, plus a set of example scenarios demonstrating them.

**What already exists (reusable as-is):**
- `QueuedResource` — queue + worker composition (the foundation for service centers)
- `Server` — concurrency + service time on top of QueuedResource
- `Resource` — shared contended capacity with acquire/release
- `Source` with `Profile` — time-varying arrival rates (constant, Poisson, custom)
- `FIFOQueue`, `LIFOQueue`, `PriorityQueue` — pluggable queue policies
- `Sink`, `Counter`, `LatencyTracker` — collectors and metrics
- `Data`, `Probe`, `BucketedData` — instrumentation
- `ConstantLatency`, `ExponentialLatency` — service time distributions
- `RandomRouter`, `LoadBalancer` — routing

**What's genuinely new (not covered):** balking, reneging, batch processing, inspection/branching, conveyor transport, shift-based capacity, machine breakdowns, inventory management, appointment scheduling.

---

## Phase A: Industrial Components

New package: `happysimulator/components/industrial/`

### 1. `BalkingQueue` — `balking.py`
A `QueuePolicy` decorator that wraps any inner policy. On `push()`, checks queue depth against a threshold and probabilistically rejects arrivals (simulating customers who see a long line and leave). Tracks `balked` count. Constructor: `BalkingQueue(inner, balk_threshold=5, balk_probability=1.0)`.

### 2. `RenegingQueuedResource` — `reneging.py`
Abstract `QueuedResource` subclass where items expire based on patience. When an item is dequeued, checks if `(now - created_at) > patience`; if so, routes to a `reneged_target` instead of processing. Patience comes from `event.context["patience_s"]` or a default. Subclasses implement `_handle_served_event()`. Tracks served/reneged counts.

### 3. `ConveyorBelt` — `conveyor.py`
Simple `Entity` that models fixed transit time between stations. Receives events, yields `transit_time`, then forwards to `downstream`. Optional capacity limit. Tracks `items_transported`, `items_in_transit`.

### 4. `InspectionStation` — `inspection.py`
`QueuedResource` that performs probabilistic pass/fail routing. Each item is inspected (yields `inspection_time`), then routed to `pass_target` with probability `pass_rate` or `fail_target` otherwise. Tracks passed/failed counts.

### 5. `BatchProcessor` — `batch_processor.py`
`Entity` that accumulates items until `batch_size` is reached or `timeout_s` expires, then processes the entire batch with a single `process_time` delay and forwards to `downstream`. Tracks batches processed, items processed, timeouts.

### 6. `ShiftSchedule` + `ShiftedServer` — `shift_schedule.py`
`Shift` dataclass defines `(start_s, end_s, capacity)`. `ShiftSchedule` is a collection of shifts with `capacity_at(time_s)`. `ShiftedServer` extends `QueuedResource` — uses the schedule to dynamically adjust concurrency. Schedules self-check events at shift boundaries.

### 7. `BreakdownScheduler` — `breakdown.py`
`Entity` that schedules random breakdowns for a target. Alternates between UP (exponential time-to-failure) and DOWN (repair time distribution) states. During DOWN, sets a `_broken` flag on the target that `has_capacity()` checks. Tracks breakdown count, total downtime, availability.

### 8. `InventoryBuffer` — `inventory.py`
`Entity` managing a stock counter with `(s, Q)` reorder policy. "Consume" events decrement stock. When `stock <= reorder_point`, triggers a replenishment order to a `supplier` entity with configurable `lead_time`. Tracks stockouts, reorder count, fill rate.

### 9. `AppointmentScheduler` — `appointment.py`
Specialized `Source`-like entity that generates arrivals at fixed appointment times with configurable `no_show_rate`. Supports combining with a separate Poisson source for walk-ins.

Each component gets a unit test file in `tests/unit/components/industrial/`.

---

## Phase B: Example Scenarios (10 examples)

Each example follows the established pattern: module docstring with ASCII architecture diagram, configuration dataclass, custom entities, `run_*_simulation()` function, `visualize_results()`, `print_summary()`, CLI with argparse.

### 1. `examples/bank_branch.py` — Bank Branch
Multiple teller windows behind a single FIFO queue. Different customer types (quick deposit ~2min, account inquiry ~5min, loan ~15min). Time-varying arrivals (morning ramp, lunch rush, afternoon taper). Customer balking (leave if queue > 10) and reneging (patience ~10min). Third teller opens at peak hours via shift schedule.
**Components**: BalkingQueue, RenegingQueuedResource, ShiftSchedule, Source.poisson, PriorityQueue, Sink

### 2. `examples/manufacturing_line.py` — Assembly Line
Four-stage pipeline: Cut → Assemble → Inspect → Package. Conveyor belts between stages. Quality inspection with 5% defect rate routing to rework loop (back to Assemble). Machine breakdowns on stations. Batch packaging (accumulate 12 items, box together).
**Components**: QueuedResource, InspectionStation, ConveyorBelt, BreakdownScheduler, BatchProcessor

### 3. `examples/hospital_er.py` — Emergency Room
Triage nurse assigns priority (1-5). Priority queue for treatment rooms. Shared resources: 3 doctors, 6 nurses, 10 beds, 1 CT scanner. Different treatment paths by severity. Critical patients preempt minor ones.
**Components**: PriorityQueue, Resource, QueuedResource, Source.poisson

### 4. `examples/call_center.py` — Call Center
IVR menu (fixed 30s delay), then skill-based routing to Sales/Support/Billing queues. Customer abandonment (reneging with exponential patience). Shift changes (morning: 8 agents, afternoon: 12, evening: 4). Service level tracking (% answered within 60s).
**Components**: RenegingQueuedResource, ShiftSchedule, Source.with_profile

### 5. `examples/grocery_store.py` — Grocery Store Checkout
Regular lanes (4), express lane (1, max 15 items), self-checkout (6). Customers choose shortest queue. Express lane eligibility based on item count. Self-checkout occasional jams (breakdown). Dynamic lane opening based on queue depth. Customer balking.
**Components**: BalkingQueue, BreakdownScheduler, ShiftSchedule

### 6. `examples/car_wash.py` — Sequential Car Wash
Pipeline: Pre-Rinse → Wash → Rinse → Dry. Different service tiers (Basic: 2 stages, Standard: 3, Premium: 4+wax). Single car per stage. Conveyor between stages. Revenue tracking by tier.
**Components**: QueuedResource, ConveyorBelt

### 7. `examples/restaurant.py` — Restaurant Simulation
Tables as Resource (15 two-tops, 5 four-tops). Reservations (appointments) + walk-ins (Poisson). Walk-ins renege after 20min wait. Kitchen pipeline: Prep → Cook → Plate. Table turnover drives throughput.
**Components**: Resource, AppointmentScheduler, RenegingQueuedResource

### 8. `examples/supply_chain.py` — Multi-Echelon Supply Chain
Three tiers: Factory → Distributor → Retailer. (s,Q) reorder policy at each level. Stochastic demand and lead times. Demonstrates bullwhip effect (variance amplification upstream). Stockout and fill rate tracking.
**Components**: InventoryBuffer, ConveyorBelt (as lead time), Source.poisson

### 9. `examples/warehouse_fulfillment.py` — Warehouse Order Fulfillment
Pick → Pack → Ship pipeline. Batch picking (accumulate 10 orders, pick together). Shared resources: pickers, pack stations, shipping dock. Zone-based picking with different walk times.
**Components**: BatchProcessor, Resource, QueuedResource

### 10. `examples/parking_lot.py` — Parking Lot
Parking spots as Resource (100 regular, 20 premium). Time-varying arrivals (morning in, evening out). Finite capacity with gate queueing. Customer balking and reneging when lot is full. Revenue by duration.
**Components**: Resource, BalkingQueue, RenegingQueuedResource, Source.with_profile

---

## Phase C: Integration & Exports

1. Integration tests for 5 key examples: `tests/integration/test_bank_branch.py`, `test_manufacturing_line.py`, `test_car_wash.py`, `test_call_center.py`, `test_grocery_store.py`
2. Update `happysimulator/components/__init__.py` to export all industrial components
3. Update `happysimulator/__init__.py` to include industrial exports
4. Update `CLAUDE.md` with industrial components section

---

## Implementation Order

Build components first (dependencies flow downward), then examples that exercise them:

1. `ConveyorBelt` (simplest, no dependencies)
2. `BalkingQueue` (pure QueuePolicy wrapper)
3. `InspectionStation` (extends QueuedResource)
4. `BatchProcessor` (standalone Entity)
5. `RenegingQueuedResource` (abstract QueuedResource subclass)
6. `ShiftSchedule` + `ShiftedServer` (extends QueuedResource)
7. `BreakdownScheduler` (wraps target Entity)
8. `InventoryBuffer` (standalone Entity)
9. `AppointmentScheduler` (Source-like Entity)
10. Examples in order: car_wash → bank_branch → manufacturing_line → call_center → grocery_store → hospital_er → parking_lot → restaurant → supply_chain → warehouse_fulfillment
11. Integration tests
12. Export updates and CLAUDE.md

---

## Key Files to Modify

- **New**: `happysimulator/components/industrial/__init__.py` + 9 component files
- **New**: `tests/unit/components/industrial/` — 9 test files
- **New**: `examples/` — 10 example files
- **New**: `tests/integration/` — 5 integration test files
- **Modify**: `happysimulator/components/__init__.py` — add industrial exports
- **Modify**: `happysimulator/__init__.py` — add industrial exports
- **Modify**: `CLAUDE.md` — add industrial components documentation

## Verification

- `pytest tests/unit/components/industrial/ -q` — all unit tests pass
- `pytest tests/integration/test_bank_branch.py tests/integration/test_manufacturing_line.py tests/integration/test_car_wash.py -q` — integration tests pass
- `python examples/bank_branch.py` — runs and prints summary
- `python examples/manufacturing_line.py` — runs and prints summary
- `python examples/car_wash.py` — runs and prints summary
- `pytest -q` — full suite still passes (no regressions)
