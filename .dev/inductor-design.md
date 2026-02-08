# Digital Inductor: Configuration-Free Burst Suppression

## Context

All existing rate limiters (TokenBucket, LeakyBucket, SlidingWindow, FixedWindow, Adaptive) require a **max throughput** setting. The digital inductor is a fundamentally different approach: it has no throughput cap but instead resists rapid *changes* in rate, just like an electrical inductor resists changes in current. This makes it ideal for passively suppressing bursts throughout a system with zero ongoing configuration.

The single parameter is a **time constant** (seconds) that controls how strongly the inductor resists rate changes. Higher = more smoothing.

## Algorithm: EWMA of Inter-Arrival Times

**O(1) state** (4 scalars):
- `_smoothed_interval` - Exponentially Weighted Moving Average of inter-arrival time
- `_last_arrival_time` - timestamp of most recent arrival
- `_last_output_time` - timestamp of most recent forwarded event
- `_poll_scheduled` - whether a drain poll is pending

**On each new arrival:**
1. Compute `dt = now - last_arrival_time`
2. Update EWMA: `alpha = 1 - exp(-dt / tau)`, then `smoothed_interval = alpha * dt + (1 - alpha) * smoothed_interval`
3. If `now - last_output_time >= smoothed_interval` → forward immediately
4. Otherwise → queue the event, schedule a drain poll

**Key property of the adaptive alpha**: `alpha = 1 - exp(-dt / tau)` makes the smoothing naturally time-aware. Short inter-arrival gaps (bursts) get low alpha (heavy smoothing). Long gaps get high alpha (fast adaptation). This is the inductor behavior.

**Why a custom Entity, not a RateLimiterPolicy**: The `RateLimiterPolicy.try_acquire()` protocol doesn't distinguish new arrivals from poll retries. The inductor must update its rate estimate only on genuine arrivals, so it needs its own Entity implementation (following the same patterns as `RateLimitedEntity`).

## Implementation Plan

### Step 1: Create `happysimulator/components/rate_limiter/inductor.py`

New `Inductor` Entity class with:

```
class InductorStats:  # dataclass
    received, forwarded, queued, dropped

class Inductor(Entity):
    __init__(name, downstream, time_constant, queue_capacity=10000)

    # O(1) state
    _smoothed_interval: float | None
    _last_arrival_time: Instant | None
    _last_output_time: Instant | None
    _poll_scheduled: bool

    # Observability
    stats: InductorStats
    received_times: list[Instant]
    forwarded_times: list[Instant]
    dropped_times: list[Instant]
    rate_history: list[tuple[Instant, float]]  # (time, estimated_rate)

    # Properties
    estimated_rate -> float  # current 1/smoothed_interval
    queue_depth -> int
    time_constant -> float

    # Event handling
    handle_event()  # dispatches arrival vs poll
    _handle_arrival()  # update EWMA + forward or queue
    _handle_poll()  # drain queue at smoothed rate
    _update_rate_estimate()  # EWMA math
    _can_forward() -> bool  # check smoothed interval
    _forward()  # create forward event to downstream
    _ensure_poll_scheduled()  # schedule drain poll
```

Follows the exact same forwarding/poll/queue patterns as `RateLimitedEntity` (lines 83-162 of `rate_limited_entity.py`), but with inductor-specific rate estimation in `_handle_arrival`.

### Step 2: Update exports

- `happysimulator/components/rate_limiter/__init__.py` - add `Inductor`, `InductorStats`
- `happysimulator/components/__init__.py` - add `Inductor`, `InductorStats`
- `happysimulator/__init__.py` - add `Inductor`, `InductorStats`

### Step 3: Unit tests - `tests/unit/test_inductor.py`

- First event always forwarded
- Steady-state constant rate passes through after warmup
- Burst events get queued (not dropped)
- Rate estimate converges to actual rate
- Higher time constant = slower adaptation
- Queue overflow → drops (bounded queue)

### Step 4: Integration test - `tests/integration/test_inductor.py`

Custom profiles defined locally in the test file (following existing pattern):

| Profile | Purpose |
|---------|---------|
| `ConstantRateProfile(rate=10)` | Steady state - inductor should track perfectly |
| `LinearRampProfile(5→50 over 60s)` | Gradual change - inductor should follow smoothly |
| `StepRateProfile(10→50 at t=20)` | Sudden burst - inductor should smooth the transition |
| `PeriodicBurstProfile(base=10, burst=80, period=10s, burst_duration=2s)` | Repeated spikes - key burst suppression scenario |

**Test functions:**

1. **`test_inductor_burst_suppression`** - Parametrized over profiles. For each profile, run an inductor (tau=2s) and plot:
   - Input rate (binned received events)
   - Output rate (binned forwarded events)
   - Inductor's estimated rate (from `rate_history`) - *the key visualization*
   - Load profile line

2. **`test_inductor_time_constants`** - Same burst profile, three inductors (tau=1, 3, 10). Shows how time constant affects smoothing.

3. **`test_inductor_vs_rate_limiters_comparison`** - Grand comparison under periodic bursts:
   - TokenBucket (capacity=20, refill=10)
   - LeakyBucket (rate=10)
   - Inductor (tau=2)
   - Inductor (tau=5)

   Key insight to visualize: traditional limiters hard-cap at their configured rate and queue/drop the rest. The inductor has no cap - it smooths the transition and eventually lets everything through at the sustained rate.

4. **`test_inductor_vs_rate_limiters_ramp`** - Same comparison but under linear ramp. Shows that the inductor naturally follows the increasing rate (no cap), while traditional limiters flatline at their configured maximum.

Each test produces PNG plots + CSV data in `test_output/`.

## Files to Create/Modify

| File | Action |
|------|--------|
| `happysimulator/components/rate_limiter/inductor.py` | **Create** - Inductor Entity + InductorStats |
| `happysimulator/components/rate_limiter/__init__.py` | Modify - add exports |
| `happysimulator/components/__init__.py` | Modify - add exports |
| `happysimulator/__init__.py` | Modify - add exports |
| `tests/unit/test_inductor.py` | **Create** - unit tests |
| `tests/integration/test_inductor.py` | **Create** - visualization + comparison tests |

## Verification

1. `pytest tests/unit/test_inductor.py -v` - all unit tests pass
2. `pytest tests/integration/test_inductor.py -v` - all integration tests pass, generates plots
3. Review generated plots in `test_output/` to verify:
   - Inductor rate_history shows smooth tracking of input rate
   - Burst suppression is visible (output rate changes more slowly than input)
   - Higher time constants produce more smoothing
   - Comparison plots clearly show inductor vs traditional rate limiter behavior differences
4. `pytest -q` - full suite passes (no regressions)
