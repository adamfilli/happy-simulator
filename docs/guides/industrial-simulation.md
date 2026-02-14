# Industrial Simulation

Components for operations research, manufacturing, healthcare, and service systems.

```python
from happysimulator.components.industrial import (
    BalkingQueue, RenegingQueuedResource,
    ConveyorBelt, InspectionStation, BatchProcessor,
    ShiftSchedule, ShiftedServer, Shift,
    BreakdownScheduler, InventoryBuffer, AppointmentScheduler,
    ConditionalRouter, PerishableInventory, PooledCycleResource,
    GateController, SplitMerge, PreemptibleResource, PreemptibleGrant,
)
```

## Component Catalog

| Component | Description |
|-----------|-------------|
| `BalkingQueue` | Wraps a queue policy; rejects arrivals when depth >= threshold |
| `RenegingQueuedResource` | Abstract; checks patience on dequeue, routes expired items |
| `ConveyorBelt` | Fixed transit time between stations, optional capacity |
| `InspectionStation` | Probabilistic pass/fail routing with configurable inspection time |
| `BatchProcessor` | Accumulates items until `batch_size` or `timeout_s`, processes as batch |
| `ShiftSchedule` + `ShiftedServer` | Time-varying capacity via shift definitions |
| `BreakdownScheduler` | Random UP/DOWN cycles on a target entity |
| `InventoryBuffer` | `(s, Q)` reorder policy with lead time |
| `AppointmentScheduler` | Fixed-time arrivals with no-show rate |
| `ConditionalRouter` | Declarative routing via ordered `(predicate, target)` list |
| `PerishableInventory` | Inventory with shelf life and periodic spoilage sweeps |
| `PooledCycleResource` | Pool of N identical units with fixed cycle time |
| `GateController` | Opens/closes on schedule or programmatically |
| `SplitMerge` | Fan-out to N targets, wait for all, merge results |
| `PreemptibleResource` | Priority-based resource with preemption support |

## Composition Pattern

Industrial simulations are built by composing these components into pipelines:

```python
from happysimulator import Simulation, Source, Sink, Instant
from happysimulator.components.industrial import (
    ConveyorBelt, InspectionStation, BatchProcessor,
)

sink = Sink()
inspector = InspectionStation(
    name="QC", inspection_time=0.5, pass_rate=0.95,
    pass_target=sink, fail_target=sink,
)
belt = ConveyorBelt(name="Belt", transit_time=2.0, downstream=inspector)
source = Source.constant(rate=1, target=belt)

sim = Simulation(
    entities=[source, belt, inspector, sink],
    end_time=Instant.from_seconds(100),
)
sim.run()
```

## Shift Schedules

```python
from happysimulator.components.industrial import Shift, ShiftSchedule, ShiftedServer

schedule = ShiftSchedule(shifts=[
    Shift(start_s=0, end_s=28800, capacity=3),      # morning: 3 workers
    Shift(start_s=28800, end_s=57600, capacity=5),   # afternoon: 5 workers
    Shift(start_s=57600, end_s=86400, capacity=2),   # night: 2 workers
])
```

## Examples

20 industrial simulations in `examples/industrial/`:

`bank_branch.py`, `manufacturing_line.py`, `hospital_er.py`, `call_center.py`, `grocery_store.py`, `car_wash.py`, `restaurant.py`, `supply_chain.py`, `warehouse_fulfillment.py`, `parking_lot.py`, `coffee_shop.py`, `drive_through.py`, `laundromat.py`, `pharmacy.py`, `theme_park.py`, `airport_terminal.py`, `hotel_operations.py`, `blood_bank.py`, `elevator_system.py`, `urgent_care.py`

## Next Steps

- [Behavioral Modeling](behavioral-modeling.md) — human agents and populations
- [Examples: Industrial](../examples/industrial.md) — full example listings
