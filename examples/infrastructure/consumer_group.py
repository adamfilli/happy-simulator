"""Consumer group: coordinated consumption with rebalancing.

Demonstrates producers writing to a partitioned log while consumers
join and leave a consumer group, triggering rebalancing.

## Architecture

```
  Producer-0 ──┐                  ┌── Consumer-0
  Producer-1 ──┼──► EventLog ◄──┤
               │    (4 parts)     ├── Consumer-1
               └                  └── Consumer-2 (joins mid-sim)
                                       ConsumerGroup
```

## Key Observations

- Rebalancing redistributes partitions when consumers join or leave.
- Consumer lag measures distance between high watermark and committed offset.
- Different assignment strategies (Range, RoundRobin, Sticky) produce
  different partition distributions and rebalance churn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from happysimulator import (
    Entity,
    Event,
    Instant,
    SimFuture,
    Simulation,
    Source,
)
from happysimulator.components.streaming.consumer_group import (
    ConsumerGroup,
    RangeAssignment,
    RoundRobinAssignment,
    StickyAssignment,
)
from happysimulator.components.streaming.event_log import EventLog

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Client entities
# =============================================================================


class Producer(Entity):
    """Writes records to the event log."""

    def __init__(self, name: str, log: EventLog):
        super().__init__(name)
        self.log = log
        self._count = 0

    def handle_event(
        self, event: Event
    ) -> Generator[float | SimFuture | tuple[float, list[Event]]]:
        self._count += 1
        key = f"user-{self._count % 20}"
        reply = SimFuture()
        yield (
            0.0,
            [
                Event(
                    time=self.now,
                    event_type="Append",
                    target=self.log,
                    context={"key": key, "value": {"seq": self._count}, "reply_future": reply},
                )
            ],
        )
        yield reply


class Consumer(Entity):
    """Polls records and commits offsets."""

    def __init__(self, name: str, group: ConsumerGroup):
        super().__init__(name)
        self.group = group
        self.records_consumed: int = 0
        self.lag_history: list[tuple[float, int]] = []

    def handle_event(
        self, event: Event
    ) -> Generator[float | SimFuture | tuple[float, list[Event]]]:
        if event.event_type == "PollCycle":
            # Poll
            reply = SimFuture()
            yield (
                0.0,
                [
                    Event(
                        time=self.now,
                        event_type="Poll",
                        target=self.group,
                        context={
                            "consumer_name": self.name,
                            "max_records": 50,
                            "reply_future": reply,
                        },
                    )
                ],
            )
            records = yield reply

            if records:
                self.records_consumed += len(records)
                # Commit max offset per partition
                offsets: dict[int, int] = {}
                for rec in records:
                    pid = rec.partition
                    if pid not in offsets or rec.offset + 1 > offsets[pid]:
                        offsets[pid] = rec.offset + 1

                yield (
                    0.0,
                    [
                        Event(
                            time=self.now,
                            event_type="Commit",
                            target=self.group,
                            context={"consumer_name": self.name, "offsets": offsets},
                        )
                    ],
                )

            # Track lag
            lag = self.group.consumer_lag(self.name)
            total_lag = sum(lag.values())
            self.lag_history.append((self.now.to_seconds(), total_lag))


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class GroupResult:
    """Results from a consumer group run."""

    strategy_name: str
    log: EventLog
    group: ConsumerGroup
    consumers: list[Consumer]


def run_consumer_group(
    strategy_name: str = "range",
    *,
    num_partitions: int = 4,
    duration_s: float = 20.0,
    produce_rate: float = 100.0,
    seed: int = 42,
) -> GroupResult:
    """Run a consumer group simulation."""
    random.seed(seed)

    strategies = {
        "range": RangeAssignment(),
        "roundrobin": RoundRobinAssignment(),
        "sticky": StickyAssignment(),
    }
    strategy = strategies[strategy_name]

    log = EventLog(name="log", num_partitions=num_partitions)
    group = ConsumerGroup(
        name="group",
        event_log=log,
        assignment_strategy=strategy,
        rebalance_delay=0.1,
        poll_latency=0.001,
    )

    producers = [Producer(f"producer-{i}", log) for i in range(2)]
    consumers = [Consumer(f"consumer-{i}", group) for i in range(3)]

    # Schedule consumer joins (c0 and c1 join early, c2 joins mid-sim)
    join_events = []
    for i, c in enumerate(consumers[:2]):
        join_events.append(
            Event(
                time=Instant.from_seconds(0.1 + i * 0.05),
                event_type="Join",
                target=group,
                context={
                    "consumer_name": c.name,
                    "consumer_entity": c,
                    "reply_future": SimFuture(),
                },
            )
        )

    # Consumer 2 joins at 1/3 of duration
    join_events.append(
        Event(
            time=Instant.from_seconds(duration_s / 3),
            event_type="Join",
            target=group,
            context={
                "consumer_name": consumers[2].name,
                "consumer_entity": consumers[2],
                "reply_future": SimFuture(),
            },
        )
    )

    # Consumer 1 leaves at 2/3 of duration
    join_events.append(
        Event(
            time=Instant.from_seconds(2 * duration_s / 3),
            event_type="Leave",
            target=group,
            context={"consumer_name": consumers[1].name, "reply_future": SimFuture()},
        )
    )

    # Poll cycles for all consumers
    poll_events = []
    poll_interval = 0.5
    for c in consumers:
        t = 1.0
        while t < duration_s:
            poll_events.append(
                Event(
                    time=Instant.from_seconds(t),
                    event_type="PollCycle",
                    target=c,
                )
            )
            t += poll_interval

    producer_sources = [
        Source.constant(
            rate=produce_rate,
            target=p,
            event_type="Produce",
            stop_after=duration_s,
        )
        for p in producers
    ]

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 5.0,
        sources=producer_sources,
        entities=[log, group, *producers, *consumers],
    )
    for e in join_events + poll_events:
        sim.schedule(e)
    sim.run()

    return GroupResult(
        strategy_name=strategy_name,
        log=log,
        group=group,
        consumers=consumers,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: list[GroupResult]) -> None:
    """Print consumer group comparison."""
    print("\n" + "=" * 70)
    print("CONSUMER GROUP — ASSIGNMENT STRATEGY COMPARISON")
    print("=" * 70)

    for r in results:
        print(f"\n  Strategy: {r.strategy_name}")
        print(f"  Rebalances: {r.group.stats.rebalances}")
        print(f"  Total records produced: {r.log.stats.records_appended:,}")
        print(f"  Total records polled:   {r.group.stats.records_polled:,}")
        print(f"  Final lag:              {r.group.total_lag():,}")

        print(f"\n  {'Consumer':>12s} {'Consumed':>9s} {'FinalLag':>9s} {'Partitions':>12s}")
        print(f"  {'-' * 46}")
        for c in r.consumers:
            lag = r.group.consumer_lag(c.name)
            total_lag = sum(lag.values()) if lag else 0
            assigned = r.group.assignments.get(c.name, [])
            parts_str = str(assigned) if assigned else "(left)"
            print(f"  {c.name:>12s} {c.records_consumed:>9d} {total_lag:>9d} {parts_str:>12s}")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(results: list[GroupResult], output_dir: Path) -> None:
    """Generate consumer lag over time charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    colors = ["steelblue", "coral", "seagreen"]

    for ax, r in zip(axes, results, strict=False):
        for i, c in enumerate(r.consumers):
            if c.lag_history:
                times = [t for t, _ in c.lag_history]
                lags = [l for _, l in c.lag_history]
                ax.plot(times, lags, label=c.name, color=colors[i % len(colors)], alpha=0.8)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Consumer Lag")
        ax.set_title(f"Strategy: {r.strategy_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Consumer Group — Lag Over Time")
    fig.tight_layout()

    path = output_dir / "consumer_group.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Consumer group demo")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/consumer_group")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print("Running consumer group simulation...")

    results = []
    for strategy in ["range", "roundrobin", "sticky"]:
        print(f"  Strategy: {strategy}...")
        r = run_consumer_group(
            strategy,
            duration_s=args.duration,
            produce_rate=args.rate,
            seed=args.seed,
        )
        results.append(r)

    print_summary(results)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(results, output_dir)
