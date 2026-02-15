"""Integration test: clock skew causes lease double-claim race condition.

Demonstrates the classic distributed systems bug where clock skew causes two
nodes to both believe they hold a lease simultaneously.

Scenario:
    - A LeaseCoordinator grants time-limited leases (TTL = 2s, using true time)
    - Two LeaseNode entities compete for the lease, checking periodically
    - Node A has a slow clock (-300ms skew): holds lease longer than it should
    - Node B has a fast clock (+300ms skew): sees lease as expired earlier

The race:
    - Node A acquires at true t=0, lease expires at true t=2.0
    - Node A's local time reads 2.0 at true t=2.3 (slow) -> holds lease 300ms too long
    - Node B's local time reads 2.0 at true t=1.7 (fast) -> tries to acquire 300ms early
    - Coordinator grants to B at true t=2.0 (server-side expiry is correct)
    - Between true t=2.0 and t=2.3, both nodes believe they hold the lease

Outputs:
    - lease_race_condition.png (3-panel plot)
    - lease_race_summary.json
"""

import json
from pathlib import Path

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.node_clock import FixedSkew, NodeClock
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Duration, Instant

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


class LeaseCoordinator(Entity):
    """Central lease server that grants time-limited leases using true time.

    Tracks who holds the lease and when it expires (both in true simulation time).
    Nodes request leases; coordinator grants if no lease is active.
    """

    def __init__(self, name: str, ttl: float):
        super().__init__(name)
        self.ttl = ttl
        self.holder: str | None = None
        self.lease_expires_at: Instant = Instant.Epoch
        # Event log: (true_time, event_type, node_name)
        self.log: list[tuple[float, str, str]] = []

    def handle_event(self, event: Event) -> list[Event] | None:
        requester_name = event.context["node_name"]
        reply_target = event.context["reply_target"]

        now = self.now
        granted = False

        # Check if current lease has expired (using true time)
        if self.holder is not None and now >= self.lease_expires_at:
            self.log.append((now.to_seconds(), "expired", self.holder))
            self.holder = None

        if self.holder is None:
            # Grant the lease
            self.holder = requester_name
            self.lease_expires_at = now + Duration.from_seconds(self.ttl)
            self.log.append((now.to_seconds(), "granted", requester_name))
            granted = True
        elif self.holder == requester_name:
            # Renewal
            self.lease_expires_at = now + Duration.from_seconds(self.ttl)
            self.log.append((now.to_seconds(), "renewed", requester_name))
            granted = True
        else:
            self.log.append((now.to_seconds(), "denied", requester_name))

        return [
            Event(
                time=now,
                event_type="LeaseResponse",
                target=reply_target,
                context={
                    "granted": granted,
                    "expires_at_s": self.lease_expires_at.to_seconds() if granted else 0,
                },
            )
        ]


class LeaseNode(Entity):
    """Node that requests and holds leases, using its local clock for decisions.

    Uses local_now (skewed) to decide when its lease has expired, simulating
    a real node that only has access to its own wall clock.
    """

    def __init__(
        self,
        name: str,
        coordinator: LeaseCoordinator,
        node_clock: NodeClock,
        check_interval: float = 0.5,
    ):
        super().__init__(name)
        self.coordinator = coordinator
        self._node_clock = node_clock
        self.check_interval = check_interval
        self.holds_lease = False
        self.lease_expires_local: float = 0.0
        # Records: (true_time, local_time, holds_lease)
        self.history: list[tuple[float, float, bool]] = []

    def set_clock(self, clock):
        super().set_clock(clock)
        self._node_clock.set_clock(clock)

    @property
    def local_now(self) -> Instant:
        return self._node_clock.now

    def handle_event(self, event: Event) -> list[Event] | None:
        true_t = self.now.to_seconds()
        local_t = self.local_now.to_seconds()

        if event.event_type == "LeaseResponse":
            granted = event.context["granted"]
            if granted:
                self.holds_lease = True
                # Node computes expiry in its own local time
                # The coordinator sent expires_at in true time; the node
                # translates this to local time (which is wrong due to skew)
                expires_true = event.context["expires_at_s"]
                # Node thinks: "my local clock will read expires_true at the
                # same moment true clock reads expires_true" — but it won't.
                # This is the bug: the node assumes local time == true time.
                self.lease_expires_local = expires_true
            self.history.append((true_t, local_t, self.holds_lease))
            return self._schedule_check()

        if event.event_type == "Check":
            # Check if we believe our lease has expired (using local clock)
            if self.holds_lease and local_t >= self.lease_expires_local:
                self.holds_lease = False

            self.history.append((true_t, local_t, self.holds_lease))

            if not self.holds_lease:
                # Try to acquire
                return [
                    Event(
                        time=self.now,
                        event_type="LeaseRequest",
                        target=self.coordinator,
                        context={
                            "node_name": self.name,
                            "reply_target": self,
                        },
                    )
                ]
            return self._schedule_check()

        if event.event_type == "Start":
            self.history.append((true_t, local_t, self.holds_lease))
            return [
                Event(
                    time=self.now,
                    event_type="LeaseRequest",
                    target=self.coordinator,
                    context={
                        "node_name": self.name,
                        "reply_target": self,
                    },
                )
            ]

        return []

    def _schedule_check(self) -> list[Event]:
        next_time = self.now + Duration.from_seconds(self.check_interval)
        return [
            Event(
                time=next_time,
                event_type="Check",
                target=self,
            )
        ]


def _find_overlap_periods(
    history_a: list[tuple[float, float, bool]],
    history_b: list[tuple[float, float, bool]],
) -> list[tuple[float, float]]:
    """Find time ranges where both nodes believe they hold the lease.

    Builds a timeline of lease-holding intervals for each node (by true time),
    then finds overlaps.
    """

    def _holding_intervals(history: list[tuple[float, float, bool]]) -> list[tuple[float, float]]:
        intervals = []
        start = None
        for true_t, _, holds in history:
            if holds and start is None:
                start = true_t
            elif not holds and start is not None:
                intervals.append((start, true_t))
                start = None
        if start is not None:
            intervals.append((start, history[-1][0]))
        return intervals

    intervals_a = _holding_intervals(history_a)
    intervals_b = _holding_intervals(history_b)

    overlaps = []
    for a_start, a_end in intervals_a:
        for b_start, b_end in intervals_b:
            overlap_start = max(a_start, b_start)
            overlap_end = min(a_end, b_end)
            if overlap_start < overlap_end:
                overlaps.append((overlap_start, overlap_end))
    return overlaps


def _run_lease_simulation(
    skew_a_s: float,
    skew_b_s: float,
    duration: float = 10.0,
    ttl: float = 2.0,
    check_interval: float = 0.3,
):
    """Run the lease simulation with given skew values.

    Returns (coordinator, node_a, node_b).
    """
    coordinator = LeaseCoordinator("Coordinator", ttl=ttl)

    clock_a = NodeClock(FixedSkew(Duration.from_seconds(skew_a_s)))
    clock_b = NodeClock(FixedSkew(Duration.from_seconds(skew_b_s)))

    node_a = LeaseNode("NodeA", coordinator, clock_a, check_interval=check_interval)
    node_b = LeaseNode("NodeB", coordinator, clock_b, check_interval=check_interval)

    sim = Simulation(
        duration=duration,
        entities=[coordinator, node_a, node_b],
    )

    # Kick off both nodes
    sim.schedule(
        Event(
            time=Instant.Epoch,
            event_type="Start",
            target=node_a,
        )
    )
    sim.schedule(
        Event(
            time=Instant.from_seconds(0.1),
            event_type="Start",
            target=node_b,
        )
    )

    sim.run()
    return coordinator, node_a, node_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLeaseRaceCondition:
    """Clock skew causes a lease double-claim (split brain)."""

    def test_skew_causes_overlap(self):
        """With skew, both nodes believe they hold the lease simultaneously."""
        _coordinator, node_a, node_b = _run_lease_simulation(
            skew_a_s=-0.3,  # slow clock
            skew_b_s=+0.3,  # fast clock
        )

        overlaps = _find_overlap_periods(node_a.history, node_b.history)
        assert len(overlaps) > 0, (
            "Expected at least one overlap period where both nodes believe they hold the lease"
        )

        # The overlap should be non-trivial
        total_overlap = sum(end - start for start, end in overlaps)
        assert total_overlap > 0.01, (
            f"Overlap of {total_overlap:.4f}s is too small to be meaningful"
        )

    def test_no_skew_no_overlap(self):
        """Without skew, no overlap occurs (control case)."""
        _coordinator, node_a, node_b = _run_lease_simulation(
            skew_a_s=0.0,
            skew_b_s=0.0,
        )

        overlaps = _find_overlap_periods(node_a.history, node_b.history)
        assert len(overlaps) == 0, f"Expected no overlap without skew, but found: {overlaps}"


class TestLeaseRaceVisualization:
    """Generates a 3-panel visualization of the lease race condition."""

    def test_visualization(self, test_output_dir: Path):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        DURATION = 10.0
        SKEW_A = -0.3
        SKEW_B = +0.3
        TTL = 2.0

        coordinator, node_a, node_b = _run_lease_simulation(
            skew_a_s=SKEW_A,
            skew_b_s=SKEW_B,
            duration=DURATION,
            ttl=TTL,
        )

        overlaps = _find_overlap_periods(node_a.history, node_b.history)

        # ── 3-panel plot ──
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Panel 1: Lease holding timeline
        ax1 = axes[0]
        for hist, y, color, label in [
            (node_a.history, 1.0, "#2196F3", f"Node A (skew={SKEW_A:+.1f}s)"),
            (node_b.history, 0.0, "#FF9800", f"Node B (skew={SKEW_B:+.1f}s)"),
        ]:
            intervals = []
            start = None
            for true_t, _, holds in hist:
                if holds and start is None:
                    start = true_t
                elif not holds and start is not None:
                    intervals.append((start, true_t))
                    start = None
            if start is not None:
                intervals.append((start, hist[-1][0]))

            for s, e in intervals:
                ax1.barh(y, e - s, left=s, height=0.4, color=color, alpha=0.7)
            ax1.barh(y, 0, color=color, alpha=0.7, label=label)  # legend entry

        # Highlight overlap regions
        for o_start, o_end in overlaps:
            ax1.axvspan(o_start, o_end, color="red", alpha=0.2)
        if overlaps:
            ax1.axvspan(0, 0, color="red", alpha=0.2, label="OVERLAP (split brain)")

        ax1.set_yticks([0.0, 1.0])
        ax1.set_yticklabels(["Node B", "Node A"])
        ax1.set_ylabel("Lease Holder")
        ax1.set_title(
            f"Lease Double-Claim Race Condition\n"
            f"TTL={TTL}s, Node A skew={SKEW_A:+.1f}s, Node B skew={SKEW_B:+.1f}s"
        )
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3, axis="x")

        # Panel 2: Local clock readings vs true time
        ax2 = axes[1]
        for hist, color, label in [
            (node_a.history, "#2196F3", "Node A local clock"),
            (node_b.history, "#FF9800", "Node B local clock"),
        ]:
            true_times = [t for t, _, _ in hist]
            local_times = [lt for _, lt, _ in hist]
            ax2.plot(true_times, local_times, color=color, linewidth=1.2, label=label)

        # True time reference line
        ax2.plot([0, DURATION], [0, DURATION], "k--", alpha=0.4, label="True time")
        ax2.set_ylabel("Local Clock Reading (s)")
        ax2.set_title("Clock Divergence")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Coordinator event log
        ax3 = axes[2]
        event_colors = {
            "granted": "#4CAF50",
            "renewed": "#8BC34A",
            "denied": "#F44336",
            "expired": "#9E9E9E",
        }
        event_markers = {
            "granted": "^",
            "renewed": "s",
            "denied": "x",
            "expired": "v",
        }
        for true_t, etype, node_name in coordinator.log:
            y = 1.0 if node_name == "NodeA" else 0.0
            ax3.scatter(
                true_t,
                y,
                color=event_colors.get(etype, "black"),
                marker=event_markers.get(etype, "o"),
                s=60,
                zorder=5,
            )

        # Legend for event types
        legend_handles = [
            mpatches.Patch(color=c, label=e.capitalize()) for e, c in event_colors.items()
        ]
        ax3.legend(handles=legend_handles, loc="upper right", fontsize=8)
        ax3.set_yticks([0.0, 1.0])
        ax3.set_yticklabels(["Node B", "Node A"])
        ax3.set_ylabel("Node")
        ax3.set_xlabel("True Simulation Time (s)")
        ax3.set_title("Coordinator Events")
        ax3.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plot_path = test_output_dir / "lease_race_condition.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000

        # ── Save summary JSON ──
        total_overlap = sum(e - s for s, e in overlaps)
        summary = {
            "config": {
                "duration_s": DURATION,
                "ttl_s": TTL,
                "skew_a_s": SKEW_A,
                "skew_b_s": SKEW_B,
            },
            "results": {
                "overlap_count": len(overlaps),
                "total_overlap_s": round(total_overlap, 6),
                "overlaps": [{"start_s": round(s, 6), "end_s": round(e, 6)} for s, e in overlaps],
                "coordinator_events": len(coordinator.log),
                "node_a_history_points": len(node_a.history),
                "node_b_history_points": len(node_b.history),
            },
        }
        json_path = test_output_dir / "lease_race_summary.json"
        with json_path.open("w") as f:
            json.dump(summary, f, indent=2)

        assert json_path.exists()
        assert len(overlaps) > 0, "Visualization should show overlaps"
