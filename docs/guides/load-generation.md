# Load Generation

`Source` generates streams of events to drive your simulation. Factory methods cover common patterns.

## Source Factories

### Constant Rate

```python
from happysimulator import Source

source = Source.constant(rate=10, target=server, event_type="Request")
```

Generates events at exactly `1/rate` second intervals. The first event arrives at `t = 1/rate`.

### Poisson (Stochastic)

```python
source = Source.poisson(rate=10, target=server, event_type="Request")
```

Exponentially distributed inter-arrival times with mean `1/rate`. Realistic for most traffic.

### Custom Profile

```python
from happysimulator import Source, ConstantRateProfile, LinearRampProfile, SpikeProfile

class RushHour(Profile):
    def rate_at(self, time_s: float) -> float:
        if 8 <= time_s <= 10:
            return 50.0  # peak
        return 10.0      # baseline

source = Source.with_profile(profile=RushHour(), target=server, poisson=True)
```

Built-in profiles:

- `ConstantRateProfile(rate)` — fixed rate
- `LinearRampProfile(start_rate, end_rate, duration_s)` — linear ramp
- `SpikeProfile(base_rate, spike_rate, spike_start, spike_duration)` — burst

### Stopping

All factories accept `stop_after` to limit generation:

```python
source = Source.constant(rate=10, target=server, stop_after=60.0)     # stop after 60s
source = Source.constant(rate=10, target=server, stop_after=Instant.from_seconds(60))
```

## Full Constructor

For advanced cases, build a `Source` from individual providers:

```python
from happysimulator import (
    Source, SimpleEventProvider, ConstantArrivalTimeProvider,
)

source = Source(
    name="Traffic",
    event_provider=SimpleEventProvider(event_type="Request", target=server),
    arrival_time_provider=ConstantArrivalTimeProvider(interval=0.1),
)
```

### Custom EventProvider

Implement `EventProvider` to control what events are generated:

```python
from happysimulator.load import EventProvider

class MyProvider(EventProvider):
    def create_event(self, time, context):
        return Event(time=time, event_type="Custom", target=self.target,
                     context={"seq": context.get("seq", 0)})
```

### Custom ArrivalTimeProvider

Control timing between events:

```python
from happysimulator.load import ArrivalTimeProvider

class BurstyArrival(ArrivalTimeProvider):
    def next_interval(self):
        # Bursts of 5, then pause
        ...
```

## Distributed Fields

`DistributedFieldProvider` adds random fields to event context:

```python
from happysimulator import DistributedFieldProvider, ZipfDistribution

provider = DistributedFieldProvider(
    inner=SimpleEventProvider(event_type="Get", target=cache),
    field="key",
    distribution=ZipfDistribution(n=1000, alpha=1.2),
)
```

## Next Steps

- [Queuing & Resources](queuing-and-resources.md) — processing pipelines with queues and shared resources
- [Core Concepts](core-concepts.md) — Event, Entity, and Simulation fundamentals
