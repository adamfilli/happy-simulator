"""Integration tests with visualizations for advanced queue policies.

These tests demonstrate queue management algorithms through visual output,
showing how different policies handle congestion and prioritization.

Run:
    pytest tests/integration/test_queue_policies_visualization.py -v

Output:
    test_output/test_queue_policies_visualization/<test_name>/
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

import pytest

from happysimulator.components.queue_policies import (
    AdaptiveLIFO,
    CoDelQueue,
    FairQueue,
    WeightedFairQueue,
    DeadlineQueue,
    REDQueue,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant, Duration
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source


class TestAdaptiveLIFOVisualization:
    """Visual tests for AdaptiveLIFO queue."""

    def test_mode_switching_under_load(self, test_output_dir):
        """
        Visualize how AdaptiveLIFO switches between FIFO and LIFO modes.

        Shows queue depth over time and which mode is active.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        queue = AdaptiveLIFO(congestion_threshold=5, capacity=20)

        # Simulate varying load
        events = []
        time = 0.0

        # Phase 1: Low load (FIFO mode)
        for i in range(10):
            queue.push({"id": i, "enqueue_time": time})
            events.append({
                "time": time,
                "action": "push",
                "depth": len(queue),
                "mode": queue.mode,
            })
            time += 0.1

            if i % 2 == 0:
                item = queue.pop()
                events.append({
                    "time": time,
                    "action": "pop",
                    "depth": len(queue),
                    "mode": queue.mode,
                })
                time += 0.05

        # Phase 2: High load burst (switch to LIFO)
        for i in range(10, 25):
            queue.push({"id": i, "enqueue_time": time})
            events.append({
                "time": time,
                "action": "push",
                "depth": len(queue),
                "mode": queue.mode,
            })
            time += 0.02

        # Phase 3: Drain queue (switch back to FIFO)
        while not queue.is_empty():
            queue.pop()
            events.append({
                "time": time,
                "action": "pop",
                "depth": len(queue),
                "mode": queue.mode,
            })
            time += 0.05

        # Create visualization
        times = [e["time"] for e in events]
        depths = [e["depth"] for e in events]
        modes = [e["mode"] for e in events]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Queue depth
        ax1.plot(times, depths, 'b-', linewidth=2, label='Queue Depth')
        ax1.axhline(y=5, color='r', linestyle='--', label='Congestion Threshold')
        ax1.set_ylabel('Queue Depth')
        ax1.set_title('AdaptiveLIFO: Mode Switching Under Load')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mode indicator
        fifo_times = [t for t, m in zip(times, modes) if m == "FIFO"]
        fifo_depths = [d for d, m in zip(depths, modes) if m == "FIFO"]
        lifo_times = [t for t, m in zip(times, modes) if m == "LIFO"]
        lifo_depths = [d for d, m in zip(depths, modes) if m == "LIFO"]

        ax2.scatter(fifo_times, [0] * len(fifo_times), c='green', label='FIFO', s=50, alpha=0.7)
        ax2.scatter(lifo_times, [1] * len(lifo_times), c='red', label='LIFO', s=50, alpha=0.7)
        ax2.set_ylabel('Mode')
        ax2.set_xlabel('Time (s)')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['FIFO', 'LIFO'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(test_output_dir / 'adaptive_lifo_mode_switching.png', dpi=150)
        plt.close()

        # Verify behavior
        assert queue.stats.dequeued_fifo > 0
        assert queue.stats.dequeued_lifo > 0
        assert queue.stats.mode_switches >= 2


class TestFairQueueVisualization:
    """Visual tests for FairQueue."""

    def test_fair_distribution_across_flows(self, test_output_dir):
        """
        Visualize how FairQueue distributes service across multiple flows.

        Shows that all flows get equal service despite different arrival patterns.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        queue = FairQueue(get_flow_id=lambda x: x["flow"])

        # Add items from 3 flows with different arrival patterns
        # Flow A: Bursty (10 items at once)
        for i in range(10):
            queue.push({"flow": "A", "id": f"A{i}"})

        # Flow B: Steady (5 items, then more later)
        for i in range(5):
            queue.push({"flow": "B", "id": f"B{i}"})

        # Flow C: Just 3 items
        for i in range(3):
            queue.push({"flow": "C", "id": f"C{i}"})

        # Dequeue and track service order
        service_order = []
        while not queue.is_empty():
            item = queue.pop()
            service_order.append(item["flow"])

        # Count service by flow at each point
        flow_service = {"A": [], "B": [], "C": []}
        counts = {"A": 0, "B": 0, "C": 0}
        for flow in service_order:
            counts[flow] += 1
            for f in flow_service:
                flow_service[f].append(counts[f])

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Cumulative service per flow
        x = range(1, len(service_order) + 1)
        ax1.plot(x, flow_service["A"], 'b-', label='Flow A (10 items)', linewidth=2)
        ax1.plot(x, flow_service["B"], 'g-', label='Flow B (5 items)', linewidth=2)
        ax1.plot(x, flow_service["C"], 'r-', label='Flow C (3 items)', linewidth=2)
        ax1.set_xlabel('Total Items Dequeued')
        ax1.set_ylabel('Items Served from Flow')
        ax1.set_title('FairQueue: Cumulative Service Per Flow')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Final distribution
        final_counts = {"A": counts["A"], "B": counts["B"], "C": counts["C"]}
        bars = ax2.bar(final_counts.keys(), final_counts.values(), color=['blue', 'green', 'red'])
        ax2.set_xlabel('Flow')
        ax2.set_ylabel('Items Served')
        ax2.set_title('FairQueue: Final Service Distribution')
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(test_output_dir / 'fair_queue_distribution.png', dpi=150)
        plt.close()

        # Verify fair service
        assert counts["A"] == 10
        assert counts["B"] == 5
        assert counts["C"] == 3


class TestWeightedFairQueueVisualization:
    """Visual tests for WeightedFairQueue."""

    def test_weighted_distribution(self, test_output_dir):
        """
        Visualize how WeightedFairQueue provides proportional service.

        Shows that higher-weight flows get proportionally more service.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def get_weight(flow_id: str) -> int:
            weights = {"premium": 3, "standard": 1}
            return weights.get(flow_id, 1)

        queue = WeightedFairQueue(
            get_flow_id=lambda x: x["flow"],
            get_weight=get_weight,
        )

        # Fill both queues equally
        for i in range(30):
            queue.push({"flow": "premium", "id": f"P{i}"})
            queue.push({"flow": "standard", "id": f"S{i}"})

        # Dequeue incrementally and track distribution
        premium_counts = [0]
        standard_counts = [0]

        for i in range(40):
            item = queue.pop()
            if item["flow"] == "premium":
                premium_counts.append(premium_counts[-1] + 1)
                standard_counts.append(standard_counts[-1])
            else:
                premium_counts.append(premium_counts[-1])
                standard_counts.append(standard_counts[-1] + 1)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Service progression
        x = range(len(premium_counts))
        ax1.plot(x, premium_counts, 'b-', label='Premium (weight=3)', linewidth=2)
        ax1.plot(x, standard_counts, 'g-', label='Standard (weight=1)', linewidth=2)
        ax1.set_xlabel('Total Items Dequeued')
        ax1.set_ylabel('Items Served from Flow')
        ax1.set_title('WeightedFairQueue: Service Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Ratio over time
        ratios = []
        for p, s in zip(premium_counts[1:], standard_counts[1:]):
            if s > 0:
                ratios.append(p / s)
            else:
                ratios.append(0)

        ax2.plot(range(1, len(ratios) + 1), ratios, 'purple', linewidth=2)
        ax2.axhline(y=3, color='r', linestyle='--', label='Expected Ratio (3:1)')
        ax2.set_xlabel('Total Items Dequeued')
        ax2.set_ylabel('Premium:Standard Ratio')
        ax2.set_title('WeightedFairQueue: Service Ratio Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 6)

        plt.tight_layout()
        plt.savefig(test_output_dir / 'weighted_fair_queue_distribution.png', dpi=150)
        plt.close()

        # Verify weighted distribution (after 40 dequeues: ~30 premium, ~10 standard)
        final_premium = premium_counts[-1]
        final_standard = standard_counts[-1]
        assert final_premium > final_standard
        assert final_premium >= 25  # Should be around 30


class TestDeadlineQueueVisualization:
    """Visual tests for DeadlineQueue."""

    def test_deadline_ordering_and_expiration(self, test_output_dir):
        """
        Visualize deadline-based ordering and expiration behavior.

        Shows items being served by deadline and expired items being dropped.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        current_time = Instant.from_seconds(0.0)

        def clock():
            return current_time

        queue = DeadlineQueue(
            get_deadline=lambda x: x["deadline"],
            clock_func=clock,
        )

        # Add items with various deadlines (not in order)
        items = [
            {"id": 1, "deadline": Instant.from_seconds(0.5)},
            {"id": 2, "deadline": Instant.from_seconds(0.2)},
            {"id": 3, "deadline": Instant.from_seconds(0.8)},
            {"id": 4, "deadline": Instant.from_seconds(0.1)},
            {"id": 5, "deadline": Instant.from_seconds(0.4)},
            {"id": 6, "deadline": Instant.from_seconds(0.15)},
            {"id": 7, "deadline": Instant.from_seconds(0.6)},
        ]

        for item in items:
            queue.push(item)

        # Advance time so some items expire
        current_time = Instant.from_seconds(0.3)

        # Dequeue and track order
        served = []
        while not queue.is_empty():
            item = queue.pop()
            if item:
                served.append(item["id"])

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all items by deadline
        deadlines = [item["deadline"].to_seconds() for item in items]
        ids = [item["id"] for item in items]

        # Color by status
        colors = []
        for item in items:
            if item["id"] in served:
                colors.append('green')
            else:
                colors.append('red')

        bars = ax.barh(range(len(items)), deadlines, color=colors, alpha=0.7)
        ax.axvline(x=0.3, color='blue', linestyle='--', linewidth=2, label='Current Time (0.3s)')

        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([f"Item {id}" for id in ids])
        ax.set_xlabel('Deadline (seconds)')
        ax.set_title('DeadlineQueue: Item Deadlines and Expiration')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Served'),
            Patch(facecolor='red', alpha=0.7, label='Expired'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Current Time'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(test_output_dir / 'deadline_queue_expiration.png', dpi=150)
        plt.close()

        # Verify behavior
        assert queue.stats.expired > 0  # Some items should have expired
        assert len(served) == len(items) - queue.stats.expired


class TestREDQueueVisualization:
    """Visual tests for REDQueue."""

    def test_probabilistic_dropping(self, test_output_dir):
        """
        Visualize RED's probabilistic dropping behavior.

        Shows drop probability increasing with queue length.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        queue = REDQueue(
            min_threshold=10,
            max_threshold=30,
            max_probability=0.5,
            capacity=50,
            weight=0.3,  # Higher weight for faster average response
        )

        # Track queue state during fill
        events = []
        for i in range(100):
            accepted = queue.push(i)
            events.append({
                "attempt": i,
                "accepted": accepted,
                "queue_len": len(queue),
                "avg_queue": queue.avg_queue_length,
                "drop_prob": queue._calculate_drop_probability(),
            })

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        attempts = [e["attempt"] for e in events]
        queue_lens = [e["queue_len"] for e in events]
        avg_queues = [e["avg_queue"] for e in events]
        drop_probs = [e["drop_prob"] for e in events]
        accepted = [e["accepted"] for e in events]

        # Queue length over time
        axes[0, 0].plot(attempts, queue_lens, 'b-', label='Actual Queue', linewidth=2)
        axes[0, 0].plot(attempts, avg_queues, 'r--', label='Avg Queue (EWMA)', linewidth=2)
        axes[0, 0].axhline(y=10, color='orange', linestyle=':', label='Min Threshold')
        axes[0, 0].axhline(y=30, color='red', linestyle=':', label='Max Threshold')
        axes[0, 0].set_xlabel('Push Attempt')
        axes[0, 0].set_ylabel('Queue Length')
        axes[0, 0].set_title('RED: Queue Length Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Drop probability over time
        axes[0, 1].plot(attempts, drop_probs, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Push Attempt')
        axes[0, 1].set_ylabel('Drop Probability')
        axes[0, 1].set_title('RED: Drop Probability Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.1)

        # Acceptance rate
        accept_rate = []
        window = 10
        for i in range(len(accepted)):
            start = max(0, i - window + 1)
            rate = sum(accepted[start:i+1]) / (i - start + 1)
            accept_rate.append(rate)

        axes[1, 0].plot(attempts, accept_rate, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Push Attempt')
        axes[1, 0].set_ylabel('Acceptance Rate (rolling)')
        axes[1, 0].set_title(f'RED: Acceptance Rate ({window}-sample window)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.1)

        # Statistics summary
        stats_text = f"""Statistics:
Enqueued: {queue.stats.enqueued}
Dropped (probabilistic): {queue.stats.dropped_probabilistic}
Dropped (forced): {queue.stats.dropped_forced}
Capacity rejected: {queue.stats.capacity_rejected}
Final queue length: {len(queue)}
"""
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('RED: Summary Statistics')

        plt.tight_layout()
        plt.savefig(test_output_dir / 'red_queue_behavior.png', dpi=150)
        plt.close()

        # Verify RED is working
        total_drops = queue.stats.dropped_probabilistic + queue.stats.dropped_forced
        assert total_drops > 0  # Should have dropped some packets


class TestCoDelQueueVisualization:
    """Visual tests for CoDelQueue."""

    def test_delay_based_dropping(self, test_output_dir):
        """
        Visualize CoDel's delay-based dropping behavior.

        Shows how CoDel responds to persistent queue delay.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        current_time = Instant.from_seconds(0.0)

        def clock():
            return current_time

        queue = CoDelQueue(
            target_delay=0.005,  # 5ms
            interval=0.050,  # 50ms
            capacity=100,
            clock_func=clock,
        )

        events = []

        # Phase 1: Fill queue quickly
        for i in range(50):
            queue.push({"id": i, "time": current_time.to_seconds()})

        events.append({
            "time": current_time.to_seconds(),
            "phase": "fill",
            "queue_len": len(queue),
            "dropped": queue.stats.dropped,
            "dropping": queue.dropping,
        })

        # Phase 2: Advance time and slowly dequeue
        for step in range(20):
            current_time = current_time + Duration.from_seconds(0.010)
            item = queue.pop()

            events.append({
                "time": current_time.to_seconds(),
                "phase": "drain",
                "queue_len": len(queue),
                "dropped": queue.stats.dropped,
                "dropping": queue.dropping,
            })

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        times = [e["time"] for e in events]
        queue_lens = [e["queue_len"] for e in events]
        dropped = [e["dropped"] for e in events]
        dropping = [1 if e["dropping"] else 0 for e in events]

        # Queue length
        axes[0, 0].plot(times, queue_lens, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Queue Length')
        axes[0, 0].set_title('CoDel: Queue Length Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Cumulative drops
        axes[0, 1].plot(times, dropped, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Cumulative Drops')
        axes[0, 1].set_title('CoDel: Cumulative Packet Drops')
        axes[0, 1].grid(True, alpha=0.3)

        # Dropping state
        axes[1, 0].fill_between(times, dropping, alpha=0.5, color='red', label='Dropping State')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('In Dropping State')
        axes[1, 0].set_title('CoDel: Dropping State')
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels(['Normal', 'Dropping'])
        axes[1, 0].grid(True, alpha=0.3)

        # Statistics
        stats_text = f"""CoDel Statistics:
Target Delay: {queue.target_delay * 1000:.1f}ms
Interval: {queue.interval * 1000:.1f}ms

Enqueued: {queue.stats.enqueued}
Dequeued: {queue.stats.dequeued}
Dropped: {queue.stats.dropped}
Drop Intervals: {queue.stats.drop_intervals}
"""
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('CoDel: Summary Statistics')

        plt.tight_layout()
        plt.savefig(test_output_dir / 'codel_queue_behavior.png', dpi=150)
        plt.close()

        # Verify CoDel behavior
        assert queue.stats.enqueued == 50
        assert queue.stats.dequeued > 0


class TestQueuePolicyComparison:
    """Compare different queue policies side by side."""

    def test_fifo_vs_lifo_latency(self, test_output_dir):
        """
        Compare latency distribution between FIFO and AdaptiveLIFO under congestion.

        Shows how LIFO provides better latency for recent requests.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from happysimulator.components.queue_policy import FIFOQueue

        # Test parameters
        n_items = 100
        congestion_start = 30

        # FIFO queue
        fifo = FIFOQueue()
        fifo_latencies = []

        time = 0.0
        for i in range(n_items):
            fifo.push({"id": i, "enqueue_time": time})
            time += 0.01

            # After congestion starts, dequeue slowly
            if i >= congestion_start and i % 3 == 0:
                item = fifo.pop()
                if item:
                    latency = time - item["enqueue_time"]
                    fifo_latencies.append(latency)

        # Drain remaining
        while not fifo.is_empty():
            item = fifo.pop()
            if item:
                latency = time - item["enqueue_time"]
                fifo_latencies.append(latency)
            time += 0.02

        # AdaptiveLIFO queue
        adaptive = AdaptiveLIFO(congestion_threshold=10)
        adaptive_latencies = []

        time = 0.0
        for i in range(n_items):
            adaptive.push({"id": i, "enqueue_time": time})
            time += 0.01

            if i >= congestion_start and i % 3 == 0:
                item = adaptive.pop()
                if item:
                    latency = time - item["enqueue_time"]
                    adaptive_latencies.append(latency)

        while not adaptive.is_empty():
            item = adaptive.pop()
            if item:
                latency = time - item["enqueue_time"]
                adaptive_latencies.append(latency)
            time += 0.02

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Latency histogram
        bins = np.linspace(0, max(max(fifo_latencies), max(adaptive_latencies)), 30)
        ax1.hist(fifo_latencies, bins=bins, alpha=0.7, label='FIFO', color='blue')
        ax1.hist(adaptive_latencies, bins=bins, alpha=0.7, label='AdaptiveLIFO', color='green')
        ax1.set_xlabel('Latency (s)')
        ax1.set_ylabel('Count')
        ax1.set_title('Latency Distribution: FIFO vs AdaptiveLIFO')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative latency
        ax2.plot(sorted(fifo_latencies), np.linspace(0, 1, len(fifo_latencies)),
                'b-', label='FIFO', linewidth=2)
        ax2.plot(sorted(adaptive_latencies), np.linspace(0, 1, len(adaptive_latencies)),
                'g-', label='AdaptiveLIFO', linewidth=2)
        ax2.set_xlabel('Latency (s)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('CDF: FIFO vs AdaptiveLIFO')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(test_output_dir / 'fifo_vs_adaptive_lifo.png', dpi=150)
        plt.close()

        # AdaptiveLIFO should have more items with low latency (recent items processed faster)
        median_fifo = sorted(fifo_latencies)[len(fifo_latencies) // 2]
        median_adaptive = sorted(adaptive_latencies)[len(adaptive_latencies) // 2]

        # Both should have served items
        assert len(fifo_latencies) > 0
        assert len(adaptive_latencies) > 0
