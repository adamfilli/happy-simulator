"""Distributed lock with fencing tokens and lease expiry.

This example demonstrates a distributed lock manager:
1. Multiple clients compete for a single named lock
2. Each grant includes a monotonically increasing fencing token
3. Leases expire automatically if not released
4. Fencing tokens prevent stale clients from corrupting resources

## Architecture Diagram

```
+------------------------------------------------------------------+
|            DISTRIBUTED LOCK WITH FENCING TOKENS                   |
+------------------------------------------------------------------+

    +-----------+       acquire("db-lock")       +------------------+
    | Client A  |------------------------------->|                  |
    +-----------+      <-- LockGrant(token=1)    |                  |
                                                 |  Distributed     |
    +-----------+       acquire("db-lock")       |  Lock Manager    |
    | Client B  |------------------------------->|                  |
    +-----------+      <-- waits (queued)        |  - Fencing       |
                                                 |    tokens        |
    +-----------+       acquire("db-lock")       |  - Lease expiry  |
    | Client C  |------------------------------->|  - FIFO waiters  |
    +-----------+      <-- waits (queued)        +------------------+

    Timeline:
    t=0.5: Client A acquires lock (token=1, lease=2s)
    t=1.0: Client B tries to acquire -> queued
    t=1.5: Client C tries to acquire -> queued
    t=2.0: Client A releases lock (token=1)
    t=2.0: Client B wakes up, gets lock (token=2)
    t=3.0: Client B's lease expires (forgot to release)
    t=3.0: Client C wakes up, gets lock (token=3)
    t=4.0: Client C releases lock (token=3)
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from happysimulator.core.entity import Entity
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.core.event import Event
from happysimulator.components.consensus.distributed_lock import (
    DistributedLock,
    DistributedLockStats,
    LockGrant,
)


# =============================================================================
# Lock Client Entity
# =============================================================================


@dataclass
class LockEvent:
    """Record of a lock lifecycle event."""
    time_s: float
    client: str
    action: str
    fencing_token: int | None = None
    detail: str = ""


class LockClient(Entity):
    """A client that acquires, holds, and releases a distributed lock.

    Args:
        name: Client identifier.
        lock_manager: The DistributedLock entity.
        lock_name: Name of the lock to acquire.
        hold_duration: How long to hold the lock before releasing.
        simulate_expiry: If True, do not release (let lease expire).
    """

    def __init__(
        self,
        name: str,
        lock_manager: DistributedLock,
        lock_name: str = "db-lock",
        hold_duration: float = 1.0,
        simulate_expiry: bool = False,
    ) -> None:
        super().__init__(name)
        self._lock_manager = lock_manager
        self._lock_name = lock_name
        self._hold_duration = hold_duration
        self._simulate_expiry = simulate_expiry
        self._grant: LockGrant | None = None
        self.events_log: list[LockEvent] = []

    def handle_event(self, event: Event):
        if event.event_type == "TryAcquire":
            return self._handle_try_acquire(event)
        if event.event_type == "LockAcquired":
            return self._handle_lock_acquired(event)
        if event.event_type == "ReleaseLock":
            return self._handle_release(event)
        return None

    def _handle_try_acquire(self, event: Event):
        """Initiate a lock acquire request."""
        now_s = self.now.to_seconds()
        self.events_log.append(LockEvent(
            time_s=now_s, client=self.name, action="acquire_request",
        ))

        # Use the lock manager's direct API
        future = self._lock_manager.acquire(self._lock_name, self.name)

        if future.is_resolved:
            # Lock was free, granted immediately
            grant = future.value
            return self._on_grant(grant)

        # Lock is held -- we need to poll for grant via event-driven callback
        # Schedule a check for when the future resolves
        return self._schedule_grant_check(future)

    def _schedule_grant_check(self, future):
        """Schedule periodic checks for lock grant resolution."""
        if future.is_resolved:
            grant = future.value
            return self._on_grant(grant)

        # Check again shortly (the lock manager will resolve the future
        # when the lock becomes available)
        def check_fn(event: Event):
            if future.is_resolved:
                grant = future.value
                return self._on_grant(grant)
            return self._schedule_grant_check(future)

        check = Event.once(
            time=self.now + 0.1,
            event_type="CheckLockGrant",
            fn=check_fn,
        )
        return [check]

    def _on_grant(self, grant: LockGrant | None):
        """Handle receiving a lock grant."""
        if grant is None:
            now_s = self.now.to_seconds()
            self.events_log.append(LockEvent(
                time_s=now_s, client=self.name, action="rejected",
            ))
            return None

        self._grant = grant
        now_s = self.now.to_seconds()
        self.events_log.append(LockEvent(
            time_s=now_s, client=self.name, action="acquired",
            fencing_token=grant.fencing_token,
            detail=f"lease={grant.lease_duration}s",
        ))

        events: list[Event] = []

        # Schedule the lease expiry event from the lock manager
        if hasattr(self._lock_manager, '_pending_expiry'):
            expiry_evt = getattr(self._lock_manager, '_pending_expiry', None)
            if expiry_evt is not None:
                events.append(expiry_evt)
                self._lock_manager._pending_expiry = None

        if not self._simulate_expiry:
            # Schedule release after hold_duration
            release_evt = Event(
                time=self.now + self._hold_duration,
                event_type="ReleaseLock",
                target=self,
                context={"fencing_token": grant.fencing_token},
            )
            events.append(release_evt)
        # else: let the lease expire naturally

        return events if events else None

    def _handle_lock_acquired(self, event: Event):
        """Handle lock acquired notification."""
        grant = event.context.get("grant")
        return self._on_grant(grant)

    def _handle_release(self, event: Event):
        """Release the lock."""
        token = event.context.get("fencing_token")
        if token is None and self._grant is not None:
            token = self._grant.fencing_token

        released = self._lock_manager.release(self._lock_name, token)
        now_s = self.now.to_seconds()
        self.events_log.append(LockEvent(
            time_s=now_s, client=self.name,
            action="released" if released else "release_failed",
            fencing_token=token,
        ))
        self._grant = None
        return None


# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the distributed lock simulation."""
    lock_manager: DistributedLock
    clients: list[LockClient]
    all_events: list[LockEvent]
    duration_s: float


# =============================================================================
# Main Simulation
# =============================================================================


def run(args=None) -> SimulationResult:
    """Run a distributed lock simulation with multiple competing clients.

    Args:
        args: Optional argparse namespace with simulation parameters.

    Returns:
        SimulationResult with lock manager state and event log.
    """
    duration_s = getattr(args, "duration", 10.0) if args else 10.0
    lease_duration = getattr(args, "lease", 2.0) if args else 2.0
    seed = getattr(args, "seed", 42) if args else 42

    if seed is not None and seed >= 0:
        random.seed(seed)

    # Create the distributed lock manager
    lock_mgr = DistributedLock(
        name="lock-manager",
        lease_duration=lease_duration,
        max_waiters=10,
    )

    # Create clients with different behaviors
    client_a = LockClient(
        name="client-A",
        lock_manager=lock_mgr,
        hold_duration=1.5,
        simulate_expiry=False,
    )
    client_b = LockClient(
        name="client-B",
        lock_manager=lock_mgr,
        hold_duration=1.0,
        simulate_expiry=True,   # will let the lease expire
    )
    client_c = LockClient(
        name="client-C",
        lock_manager=lock_mgr,
        hold_duration=0.5,
        simulate_expiry=False,
    )

    clients = [client_a, client_b, client_c]

    # Schedule acquire events at different times
    acquire_a = Event(
        time=Instant.from_seconds(0.5),
        event_type="TryAcquire",
        target=client_a,
    )
    acquire_b = Event(
        time=Instant.from_seconds(1.0),
        event_type="TryAcquire",
        target=client_b,
    )
    acquire_c = Event(
        time=Instant.from_seconds(1.5),
        event_type="TryAcquire",
        target=client_c,
    )

    # Schedule a second round of acquisitions
    acquire_a2 = Event(
        time=Instant.from_seconds(6.0),
        event_type="TryAcquire",
        target=client_a,
    )
    acquire_b2 = Event(
        time=Instant.from_seconds(6.5),
        event_type="TryAcquire",
        target=client_b,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s),
        entities=[lock_mgr, client_a, client_b, client_c],
    )
    sim.schedule(acquire_a)
    sim.schedule(acquire_b)
    sim.schedule(acquire_c)
    sim.schedule(acquire_a2)
    sim.schedule(acquire_b2)
    sim.run()

    # Collect all events from clients
    all_events = []
    for c in clients:
        all_events.extend(c.events_log)
    all_events.sort(key=lambda e: e.time_s)

    return SimulationResult(
        lock_manager=lock_mgr,
        clients=clients,
        all_events=all_events,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary of the distributed lock simulation."""
    print("\n" + "=" * 70)
    print("DISTRIBUTED LOCK WITH FENCING TOKENS RESULTS")
    print("=" * 70)

    stats = result.lock_manager.stats
    print(f"\nLock Manager Statistics:")
    print(f"  Total acquires:    {stats.total_acquires}")
    print(f"  Total releases:    {stats.total_releases}")
    print(f"  Total expirations: {stats.total_expirations}")
    print(f"  Total rejections:  {stats.total_rejections}")
    print(f"  Active locks:      {stats.active_locks}")
    print(f"  Queued waiters:    {stats.total_waiters}")

    print(f"\nEvent Timeline:")
    print(f"  {'Time':<8} {'Client':<12} {'Action':<18} {'Token':<8} {'Detail'}")
    print(f"  {'-' * 60}")

    for evt in result.all_events:
        token_str = str(evt.fencing_token) if evt.fencing_token is not None else "-"
        print(f"  {evt.time_s:<8.2f} {evt.client:<12} {evt.action:<18} "
              f"{token_str:<8} {evt.detail}")

    # Verify fencing token monotonicity
    tokens = [e.fencing_token for e in result.all_events
              if e.fencing_token is not None and e.action == "acquired"]
    print(f"\nFencing Token Analysis:")
    print(f"  Tokens granted: {tokens}")
    if tokens:
        is_monotonic = all(tokens[i] < tokens[i + 1] for i in range(len(tokens) - 1))
        print(f"  Strictly monotonic: {'YES' if is_monotonic else 'NO'}")
        print(f"  Min token: {min(tokens)}")
        print(f"  Max token: {max(tokens)}")

    # Per-client summary
    print(f"\nPer-Client Summary:")
    for client in result.clients:
        acquires = [e for e in client.events_log if e.action == "acquired"]
        releases = [e for e in client.events_log if e.action == "released"]
        print(f"  {client.name}: {len(acquires)} acquires, {len(releases)} releases")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualization of lock lifecycle and fencing tokens."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: Fencing token timeline
    ax = axes[0]
    client_colors = {"client-A": "steelblue", "client-B": "seagreen", "client-C": "indianred"}

    for evt in result.all_events:
        if evt.action == "acquired" and evt.fencing_token is not None:
            color = client_colors.get(evt.client, "gray")
            ax.scatter(evt.time_s, evt.fencing_token, color=color, s=100, zorder=5,
                       label=evt.client)
            ax.annotate(f"T={evt.fencing_token}",
                        (evt.time_s, evt.fencing_token),
                        textcoords="offset points", xytext=(5, 10), fontsize=9)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Fencing Token")
    ax.set_title("Fencing Token Progression")
    ax.grid(True, alpha=0.3)

    # Chart 2: Lock manager stats
    ax = axes[1]
    stats = result.lock_manager.stats
    categories = ["Acquires", "Releases", "Expirations", "Rejections"]
    values = [stats.total_acquires, stats.total_releases,
              stats.total_expirations, stats.total_rejections]
    colors = ["seagreen", "steelblue", "gold", "indianred"]

    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Lock Manager Statistics")
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Distributed Lock with Fencing Tokens", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "distributed_lock_fencing.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'distributed_lock_fencing.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed lock with fencing tokens")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration (s)")
    parser.add_argument("--lease", type=float, default=2.0,
                        help="Lock lease duration (s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/distributed_lock",
                        help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = None

    print("Running distributed lock simulation...")
    print(f"  Duration: {args.duration}s")
    print(f"  Lease duration: {args.lease}s")
    print(f"  Random seed: {args.seed if args.seed is not None else 'random'}")

    result = run(args)
    print_summary(result)

    if not args.no_viz:
        try:
            import matplotlib
            matplotlib.use("Agg")
            output_dir = Path(args.output)
            visualize_results(result, output_dir)
            print(f"\nVisualizations saved to: {output_dir.absolute()}")
        except ImportError:
            print("\nSkipping visualization (matplotlib not installed)")
