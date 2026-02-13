# Plan: Industrial Simulation Components & Examples — Wave 2

## Context

Wave 1 (completed in PR #54) added 9 industrial components and 10 example simulations. The user wants a second wave expanding into more non-distributed-systems domains. This plan adds 6 new reusable components and 10 new example simulations covering airports, theme parks, coffee shops, pharmacies, laundromats, hotels, blood banks, drive-throughs, elevators, and urgent care clinics.

---

## Phase A: New Components (6)

All in `happysimulator/components/industrial/`. Each gets a unit test file in `tests/unit/components/industrial/`.

### 1. `ConditionalRouter` — `conditional_router.py`
Declarative routing based on event context predicates. Replaces the hand-coded if-else routing seen in car_wash TierRouter, call_center SkillRouter, grocery_store LaneChooser, etc.

```python
class ConditionalRouter(Entity):
    def __init__(self, name, routes: list[tuple[Callable[[Event], bool], Entity]],
                 default: Entity | None = None, drop_unmatched: bool = False)
```
- `routes`: ordered list of `(predicate, target)` — first match wins
- Tracks `routed_counts: dict[str, int]`, `dropped: int`
- Factory: `ConditionalRouter.by_context_field(name, field, mapping, default)` for simple value-based dispatch
- Stats: `RouterStats(total_routed, dropped, by_target)`

### 2. `PerishableInventory` — `perishable_inventory.py`
Inventory where items have shelf lives. Periodic spoilage sweeps remove expired items.

```python
class PerishableInventory(Entity):
    def __init__(self, name, initial_stock=100, shelf_life_s=3600.0,
                 spoilage_check_interval_s=60.0, reorder_point=20, order_quantity=50,
                 lead_time=5.0, downstream=None, waste_target=None)
```
- Internal `_items: deque[tuple[Instant, int]]` — FIFO batches with arrival timestamps
- Self-perpetuating `_SpoilageCheck` events (like BreakdownScheduler pattern)
- `start_event()` returns initial spoilage check event
- Stats: `PerishableInventoryStats(current_stock, total_consumed, total_spoiled, stockouts, reorders, waste_rate)`

### 3. `PooledCycleResource` — `pooled_cycle.py`
Pool of N identical units with automatic fixed-duration release (washing machines, ride seats, rental cars).

```python
class PooledCycleResource(Entity):
    def __init__(self, name, pool_size: int, cycle_time: float,
                 downstream: Entity | None = None, queue_capacity: int = 0)
```
- Distinct from `Resource`: units are discrete (not fungible capacity), hold duration is fixed and automatic (not caller-controlled), includes integrated queue
- `handle_event()`: if unit available, start cycle; otherwise enqueue. After `yield cycle_time`, release unit and try dequeue.
- Stats: `PooledCycleStats(pool_size, available, active, queued, completed, rejected, utilization)`

### 4. `GateController` — `gate_controller.py`
Opens/closes on schedule or programmatically. When closed, queues arrivals. On open, flushes queue.

```python
class GateController(Entity):
    def __init__(self, name, downstream: Entity,
                 schedule: list[tuple[float, float]] | None = None,
                 initially_open: bool = True, queue_capacity: int = 0)
```
- `schedule`: list of `(open_at_s, close_at_s)` intervals
- `open()` / `close()` methods for programmatic control (return events)
- `start_events()` pre-schedules transitions from schedule
- Stats: `GateStats(passed_through, queued_while_closed, rejected, open_cycles)`

### 5. `SplitMerge` — `split_merge.py`
Fan-out one event to N parallel targets, wait for all to complete via `all_of`, forward merged result.

```python
class SplitMerge(Entity):
    def __init__(self, name, targets: list[Entity], downstream: Entity,
                 split_event_type="SubTask", merge_event_type="Merged")
```
- `handle_event()` generator: create N SimFutures, schedule sub-events, `yield all_of(*futures)`, forward merged result with `context["sub_results"]`
- Targets must resolve `event.context["reply_future"]` when done
- Stats: `SplitMergeStats(splits_initiated, merges_completed, fan_out)`

### 6. `PreemptibleResource` — `preemptible_resource.py`
Resource where higher-priority requests can evict lower-priority holders.

```python
class PreemptibleResource(Entity):
    def __init__(self, name, capacity: int)
```
- `acquire(amount, priority, preempt=True) -> SimFuture`: grants immediately if capacity available; preempts lowest-priority holder if `preempt=True`; otherwise queues by priority
- Extended `Grant` with `on_preempt` callback and `preempted` flag
- Evicted grant's `on_preempt` callback fires with notification
- Stats: `PreemptibleResourceStats(capacity, available, acquisitions, releases, preemptions, contentions)`

---

## Phase B: Example Simulations (10)

Each follows the established pattern: module docstring with ASCII diagram, frozen config dataclass, custom entities, `run_*_simulation()`, `print_summary()`, CLI with argparse.

### 1. `examples/coffee_shop.py` — Coffee Shop
Walk-in + mobile orders. Mobile gets priority at counter. ConditionalRouter splits by drink type (drip/espresso/blended). Drip uses BatchProcessor (brew pot of 12). Espresso/blended are individual QueuedResource stations.
**Components**: ConditionalRouter, BatchProcessor, PriorityQueue, QueuedResource

### 2. `examples/drive_through.py` — Drive-Through Restaurant
Single-lane pipeline: OrderBoard → Kitchen → PaymentWindow → PickupWindow. ConditionalRouter splits simple vs complex orders to fast/slow kitchen paths. BalkingQueue at entrance.
**Components**: ConditionalRouter, BalkingQueue, QueuedResource, ConveyorBelt

### 3. `examples/laundromat.py` — Laundromat
8 washers (35min cycle) → 6 dryers (45min cycle) → 4 folding tables (Resource, ~10min hold). Customers renege if washer wait > 15min.
**Components**: PooledCycleResource, Resource, RenegingQueuedResource

### 4. `examples/pharmacy.py` — Pharmacy
5-stage pipeline: DropOff → DataEntry → PharmacistVerify (InspectionStation, 92% pass, fail → rework) → Filling → Pickup. Controlled substances in PerishableInventory (30-day shelf life).
**Components**: PerishableInventory, InspectionStation, QueuedResource

### 5. `examples/theme_park.py` — Theme Park
3 rides as PooledCycleResource (roller coaster: 24 seats/3min, ferris wheel: 40/10min, water ride: 12/5min). ConditionalRouter for ride choice. FastPass holders get PriorityQueue. BalkingQueue per ride.
**Components**: PooledCycleResource, ConditionalRouter, BalkingQueue, PriorityQueue

### 6. `examples/airport_terminal.py` — Airport Terminal
ConditionalRouter by ticket class → Economy/Business/First check-in counters → ConveyorBelt (baggage) → Security (PriorityQueue for TSA PreCheck) → Gate lounge → Boarding (PooledCycleResource). ShiftSchedule for staffing.
**Components**: ConditionalRouter, PooledCycleResource, ShiftSchedule, ConveyorBelt, PriorityQueue

### 7. `examples/hotel_operations.py` — Hotel Operations
AppointmentScheduler (reservations) + Source (walk-ins) → FrontDesk → Rooms (Resource, 80 rooms, long hold) → CheckOut → Housekeeping (ShiftSchedule, day shift only). GateController for check-in window (no check-in 11am-3pm during room turnover).
**Components**: GateController, Resource, AppointmentScheduler, ShiftSchedule, QueuedResource

### 8. `examples/blood_bank.py` — Blood Bank
AppointmentScheduler (donors) → DonationStation → SplitMerge (fan-out to 3 parallel tests: type, infection, antibody) → ConditionalRouter (all pass?) → PerishableInventory (42-day shelf life). Separate Source for transfusion demand consuming from inventory.
**Components**: SplitMerge, PerishableInventory, ConditionalRouter, AppointmentScheduler, InspectionStation

### 9. `examples/elevator_system.py` — Elevator System
Sources per floor (different rates) → FloorQueues → ElevatorDispatcher → GateController (doors) → ElevatorCar (Resource, 8 capacity, travel time proportional to floors) → destination Sink. 3 elevators, 10 floors.
**Components**: GateController, Resource, QueuedResource, PriorityQueue

### 10. `examples/urgent_care.py` — Urgent Care Clinic
Source → Reception → Triage (assigns priority) → ConditionalRouter (critical/non-critical) → critical: PreemptibleResource (2 trauma bays, preempts minor) / non-critical: ExamRooms (QueuedResource, 4 rooms) → Treatment → Sink. Non-critical patients renege after 90min.
**Components**: PreemptibleResource, ConditionalRouter, RenegingQueuedResource, PriorityQueue

---

## Phase C: Integration & Exports

1. Update `happysimulator/components/industrial/__init__.py` — export 6 new components + stats classes
2. Update `happysimulator/components/__init__.py` — add new industrial exports
3. Update `happysimulator/__init__.py` — add new industrial exports
4. Integration tests for all 10 examples in `tests/integration/`
5. Update `CLAUDE.md` — add new components to industrial table

---

## Implementation Order

### Components (dependency order):
1. `ConditionalRouter` — simplest, no deps, unblocks 6 examples
2. `PerishableInventory` — no deps, unblocks pharmacy + blood bank
3. `PooledCycleResource` — no deps, unblocks theme park + laundromat + airport
4. `GateController` — no deps, unblocks hotel + elevator
5. `SplitMerge` — uses existing `all_of`, unblocks blood bank
6. `PreemptibleResource` — most complex, unblocks urgent care

### Examples (order of component readiness):
1. coffee_shop → 2. drive_through → 3. laundromat → 4. pharmacy → 5. theme_park → 6. airport_terminal → 7. hotel_operations → 8. blood_bank → 9. elevator_system → 10. urgent_care

### Finalization:
- Export updates, integration tests, CLAUDE.md update

---

## Key Files to Modify

- **New**: `happysimulator/components/industrial/` — 6 component files
- **New**: `tests/unit/components/industrial/` — 6 test files
- **New**: `examples/` — 10 example files
- **New**: `tests/integration/` — 10 integration test files
- **Modify**: `happysimulator/components/industrial/__init__.py`
- **Modify**: `happysimulator/components/__init__.py`
- **Modify**: `happysimulator/__init__.py`
- **Modify**: `CLAUDE.md`

## Reference Files (patterns to follow)
- `happysimulator/components/industrial/inventory.py` — PerishableInventory extends this concept
- `happysimulator/components/resource.py` — PreemptibleResource and PooledCycleResource follow this pattern
- `happysimulator/components/industrial/breakdown.py` — self-perpetuating event pattern (start_event + daemon)
- `examples/bank_branch.py` — canonical example pattern (docstring, ASCII, config, run, print, CLI)

## Verification

```bash
pytest tests/unit/components/industrial/ -q          # all unit tests pass
pytest tests/integration/ -q                          # all integration tests pass
python examples/coffee_shop.py                        # runs and prints summary
python examples/blood_bank.py                         # runs and prints summary
python examples/urgent_care.py                        # runs and prints summary
pytest -q                                             # full suite, no regressions
```
