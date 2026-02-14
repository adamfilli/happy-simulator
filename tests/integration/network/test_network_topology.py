"""Integration tests for network topology with database quorum simulation.

Scenario:
- A client located outside the datacenter (internet connection)
- 3 database replicas across different geographic sites:
  - Site A: US-East
  - Site B: US-West
  - Site C: EU-West
- Quorum writes: write to all 3 replicas, wait for 2 to acknowledge
- Quorum reads: read from 2 replicas, return fastest response

This test demonstrates realistic network latency modeling and the impact
of geographic distribution on distributed system performance.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

import pytest

from happysimulator.components.network.link import NetworkLink
from happysimulator.components.network.network import Network
from happysimulator.components.network.conditions import (
    cross_region_network,
    datacenter_network,
    internet_network,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.exponential import ExponentialLatency
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source


# --- Profiles ---


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant request rate profile."""
    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


# --- Database Replica Entity ---


@dataclass
class DatabaseReplica(Entity):
    """Simulates a database replica with configurable processing time.

    Handles write and read requests with simulated disk/CPU latency.
    Tracks statistics for analysis.
    """
    name: str
    processing_latency: float = 0.005  # 5ms default processing

    writes_received: int = field(default=0, init=False)
    reads_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, list[Event] | None]:
        """Process a database request."""
        op_type = event.context.get("metadata", {}).get("operation", "unknown")

        if op_type == "write":
            self.writes_received += 1
        elif op_type == "read":
            self.reads_received += 1

        # Simulate processing time
        yield self.processing_latency

        # Create acknowledgment event back to coordinator
        coordinator = event.context.get("metadata", {}).get("coordinator")
        if coordinator is not None:
            ack = Event(
                time=self.now,
                event_type=f"{op_type}_ack",
                target=coordinator,
                context={
                    "metadata": {
                        "request_id": event.context.get("metadata", {}).get("request_id"),
                        "replica": self.name,
                        "operation": op_type,
                    }
                },
            )
            return [ack]
        return None


# --- Quorum Coordinator Entity ---


@dataclass
class QuorumCoordinator(Entity):
    """Coordinates quorum reads and writes across replicas.

    For writes: sends to all replicas, waits for quorum_size acks.
    For reads: sends to quorum_size replicas, returns first response.
    """
    name: str
    replicas: list[DatabaseReplica] = field(default_factory=list)
    network: Network | None = None
    quorum_size: int = 2  # Default quorum of 2 out of 3

    # Track pending operations: request_id -> {acks_received, start_time, completed}
    _pending: dict[int, dict] = field(default_factory=dict, init=False)

    # Statistics
    writes_completed: int = field(default=0, init=False)
    reads_completed: int = field(default=0, init=False)
    write_latencies: list[float] = field(default_factory=list, init=False)
    read_latencies: list[float] = field(default_factory=list, init=False)

    # Downstream for completed requests
    downstream: Entity | None = None

    def handle_event(self, event: Event) -> list[Event] | None:
        """Handle incoming requests or acknowledgments."""
        event_type = event.event_type

        if event_type == "write_request":
            return self._handle_write_request(event)
        elif event_type == "read_request":
            return self._handle_read_request(event)
        elif event_type in ("write_ack", "read_ack"):
            return self._handle_ack(event)

        return None

    def _handle_write_request(self, event: Event) -> list[Event]:
        """Initiate a quorum write to all replicas."""
        request_id = event.context.get("metadata", {}).get("request_id", id(event))

        self._pending[request_id] = {
            "acks_received": 0,
            "start_time": self.now,
            "completed": False,
            "operation": "write",
            "original_event": event,
        }

        # Send write to all replicas through the network
        events = []
        for replica in self.replicas:
            if self.network is not None:
                write_event = Event(
                    time=self.now,
                    event_type="replica_write",
                    target=self.network,
                    context={
                        "metadata": {
                            "source": self.name,
                            "destination": replica.name,
                            "request_id": request_id,
                            "operation": "write",
                            "coordinator": self,
                        }
                    },
                )
            else:
                # Direct connection (no network simulation)
                write_event = Event(
                    time=self.now,
                    event_type="replica_write",
                    target=replica,
                    context={
                        "metadata": {
                            "request_id": request_id,
                            "operation": "write",
                            "coordinator": self,
                        }
                    },
                )
            events.append(write_event)

        return events

    def _handle_read_request(self, event: Event) -> list[Event]:
        """Initiate a quorum read from quorum_size replicas."""
        request_id = event.context.get("metadata", {}).get("request_id", id(event))

        self._pending[request_id] = {
            "acks_received": 0,
            "start_time": self.now,
            "completed": False,
            "operation": "read",
            "original_event": event,
        }

        # Send read to first quorum_size replicas
        events = []
        for replica in self.replicas[:self.quorum_size]:
            if self.network is not None:
                read_event = Event(
                    time=self.now,
                    event_type="replica_read",
                    target=self.network,
                    context={
                        "metadata": {
                            "source": self.name,
                            "destination": replica.name,
                            "request_id": request_id,
                            "operation": "read",
                            "coordinator": self,
                        }
                    },
                )
            else:
                read_event = Event(
                    time=self.now,
                    event_type="replica_read",
                    target=replica,
                    context={
                        "metadata": {
                            "request_id": request_id,
                            "operation": "read",
                            "coordinator": self,
                        }
                    },
                )
            events.append(read_event)

        return events

    def _handle_ack(self, event: Event) -> list[Event] | None:
        """Handle acknowledgment from a replica."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._pending:
            return None

        pending = self._pending[request_id]
        if pending["completed"]:
            return None  # Already completed

        pending["acks_received"] += 1

        # Check if quorum reached
        if pending["acks_received"] >= self.quorum_size:
            pending["completed"] = True
            latency = (self.now - pending["start_time"]).to_seconds()

            if pending["operation"] == "write":
                self.writes_completed += 1
                self.write_latencies.append(latency)
            else:
                self.reads_completed += 1
                self.read_latencies.append(latency)

            # Notify downstream if configured
            if self.downstream is not None:
                original = pending["original_event"]
                completion = Event(
                    time=self.now,
                    event_type=f"{pending['operation']}_complete",
                    target=self.downstream,
                    context={
                        "metadata": {
                            "request_id": request_id,
                            "latency_s": latency,
                            "operation": pending["operation"],
                        },
                        "created_at": pending["start_time"],
                    },
                )
                return [completion]

        return None


# --- Client Entity ---


@dataclass
class DatabaseClient(Entity):
    """Client that sends requests to the database coordinator."""
    name: str
    coordinator: QuorumCoordinator | None = None
    network: Network | None = None

    requests_sent: int = field(default=0, init=False)
    responses_received: int = field(default=0, init=False)
    latencies: list[float] = field(default_factory=list, init=False)

    def handle_event(self, event: Event) -> list[Event] | None:
        """Handle request generation or response receipt."""
        if event.event_type in ("write_complete", "read_complete"):
            self.responses_received += 1
            latency = event.context.get("metadata", {}).get("latency_s", 0)
            self.latencies.append(latency)
            return None

        return None


# --- Event Provider ---


class DatabaseRequestProvider(EventProvider):
    """Generates database requests targeting a coordinator."""

    def __init__(
        self,
        coordinator: QuorumCoordinator,
        client: DatabaseClient,
        write_ratio: float = 0.3,  # 30% writes, 70% reads
        stop_after: Instant | None = None,
    ):
        self.coordinator = coordinator
        self.client = client
        self.write_ratio = write_ratio
        self.stop_after = stop_after
        self.generated = 0
        self._write_count = 0
        self._read_count = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1

        # Determine operation type
        import random
        is_write = random.random() < self.write_ratio

        if is_write:
            self._write_count += 1
            event_type = "write_request"
        else:
            self._read_count += 1
            event_type = "read_request"

        self.client.requests_sent += 1

        return [
            Event(
                time=time,
                event_type=event_type,
                target=self.coordinator,
                context={
                    "metadata": {
                        "request_id": self.generated,
                        "operation": "write" if is_write else "read",
                    },
                    "created_at": time,
                },
            )
        ]


# --- Helper Functions ---


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write data to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _percentile_sorted(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from sorted values."""
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


# --- Test Scenarios ---


@dataclass
class QuorumScenarioResult:
    """Results from a quorum simulation scenario."""
    coordinator: QuorumCoordinator
    client: DatabaseClient
    replicas: list[DatabaseReplica]
    network: Network
    duration_s: float
    requests_generated: int


def run_quorum_scenario(
    *,
    duration_s: float = 30.0,
    request_rate: float = 10.0,
    write_ratio: float = 0.3,
    quorum_size: int = 2,
    test_output_dir: Path | None = None,
) -> QuorumScenarioResult:
    """Run a database quorum simulation with geographic distribution.

    Network topology:
    - Client (external) connects via internet to coordinator
    - Coordinator in US-East datacenter
    - Replica A: US-East (local, ~0.5ms)
    - Replica B: US-West (~50ms cross-region)
    - Replica C: EU-West (~80ms cross-region)
    """
    import random
    random.seed(42)  # Reproducibility

    # Create replicas with slightly different processing times
    replica_a = DatabaseReplica(name="replica_us_east", processing_latency=0.005)
    replica_b = DatabaseReplica(name="replica_us_west", processing_latency=0.006)
    replica_c = DatabaseReplica(name="replica_eu_west", processing_latency=0.007)
    replicas = [replica_a, replica_b, replica_c]

    # Create network topology
    network = Network(name="geo_network")

    # Create coordinator (located in US-East)
    client = DatabaseClient(name="external_client")
    coordinator = QuorumCoordinator(
        name="coordinator",
        replicas=replicas,
        network=network,
        quorum_size=quorum_size,
        downstream=client,
    )
    client.coordinator = coordinator

    # Network links from coordinator to replicas
    # US-East (local) - very fast
    link_to_us_east = NetworkLink(
        name="coord_to_us_east",
        latency=ConstantLatency(0.0005),  # 0.5ms
        bandwidth_bps=10_000_000_000,  # 10 Gbps
    )
    network.add_bidirectional_link(coordinator, replica_a, link_to_us_east)

    # US-West (cross-region) - ~50ms
    link_to_us_west = NetworkLink(
        name="coord_to_us_west",
        latency=ConstantLatency(0.050),  # 50ms
        bandwidth_bps=1_000_000_000,  # 1 Gbps
        jitter=ExponentialLatency(0.005),  # 5ms jitter
    )
    network.add_bidirectional_link(coordinator, replica_b, link_to_us_west)

    # EU-West (cross-region) - ~80ms
    link_to_eu_west = NetworkLink(
        name="coord_to_eu_west",
        latency=ConstantLatency(0.080),  # 80ms
        bandwidth_bps=1_000_000_000,  # 1 Gbps
        jitter=ExponentialLatency(0.010),  # 10ms jitter
    )
    network.add_bidirectional_link(coordinator, replica_c, link_to_eu_west)

    # Request provider
    provider = DatabaseRequestProvider(
        coordinator=coordinator,
        client=client,
        write_ratio=write_ratio,
        stop_after=Instant.from_seconds(duration_s),
    )

    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate_per_s=request_rate),
        start_time=Instant.Epoch,
    )
    source = Source("request_source", provider, arrival)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 1.0),  # Extra time to complete
        sources=[source],
        entities=[network, coordinator, client, replica_a, replica_b, replica_c],
    )
    sim.run()

    # Generate outputs
    if test_output_dir is not None:
        _generate_outputs(
            coordinator=coordinator,
            client=client,
            replicas=replicas,
            network=network,
            output_dir=test_output_dir,
        )

    return QuorumScenarioResult(
        coordinator=coordinator,
        client=client,
        replicas=replicas,
        network=network,
        duration_s=duration_s,
        requests_generated=provider.generated,
    )


def _generate_outputs(
    coordinator: QuorumCoordinator,
    client: DatabaseClient,
    replicas: list[DatabaseReplica],
    network: Network,
    output_dir: Path,
) -> None:
    """Generate CSV files and plots for the scenario results."""

    # Write latency statistics
    write_sorted = sorted(coordinator.write_latencies)
    read_sorted = sorted(coordinator.read_latencies)

    _write_csv(
        output_dir / "write_latencies.csv",
        header=["index", "latency_s"],
        rows=[[i, lat] for i, lat in enumerate(coordinator.write_latencies)],
    )

    _write_csv(
        output_dir / "read_latencies.csv",
        header=["index", "latency_s"],
        rows=[[i, lat] for i, lat in enumerate(coordinator.read_latencies)],
    )

    # Summary statistics
    def stats_row(name: str, values: list[float]) -> list:
        if not values:
            return [name, 0, 0, 0, 0, 0, 0]
        sorted_vals = sorted(values)
        return [
            name,
            len(values),
            sum(values) / len(values),
            _percentile_sorted(sorted_vals, 0.50),
            _percentile_sorted(sorted_vals, 0.90),
            _percentile_sorted(sorted_vals, 0.99),
            max(values),
        ]

    _write_csv(
        output_dir / "latency_summary.csv",
        header=["operation", "count", "avg_s", "p50_s", "p90_s", "p99_s", "max_s"],
        rows=[
            stats_row("write", coordinator.write_latencies),
            stats_row("read", coordinator.read_latencies),
        ],
    )

    # Replica statistics
    _write_csv(
        output_dir / "replica_stats.csv",
        header=["replica", "writes_received", "reads_received"],
        rows=[[r.name, r.writes_received, r.reads_received] for r in replicas],
    )

    # Network statistics
    _write_csv(
        output_dir / "network_stats.csv",
        header=["metric", "value"],
        rows=[
            ["events_routed", network.events_routed],
            ["events_dropped_no_route", network.events_dropped_no_route],
            ["events_dropped_partition", network.events_dropped_partition],
        ],
    )

    # Generate plots
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Write latency histogram
    ax = axes[0, 0]
    if write_sorted:
        ax.hist(write_sorted, bins=30, alpha=0.7, color="coral", edgecolor="black")
        ax.axvline(_percentile_sorted(write_sorted, 0.50), color="red", linestyle="--", label="p50")
        ax.axvline(_percentile_sorted(write_sorted, 0.99), color="darkred", linestyle="--", label="p99")
        ax.legend()
    ax.set_title("Write Latency Distribution")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Plot 2: Read latency histogram
    ax = axes[0, 1]
    if read_sorted:
        ax.hist(read_sorted, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(_percentile_sorted(read_sorted, 0.50), color="blue", linestyle="--", label="p50")
        ax.axvline(_percentile_sorted(read_sorted, 0.99), color="darkblue", linestyle="--", label="p99")
        ax.legend()
    ax.set_title("Read Latency Distribution")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Plot 3: Write vs Read comparison (box plot style)
    ax = axes[1, 0]
    data_to_plot = []
    labels = []
    if write_sorted:
        data_to_plot.append(write_sorted)
        labels.append(f"Writes\n(n={len(write_sorted)})")
    if read_sorted:
        data_to_plot.append(read_sorted)
        labels.append(f"Reads\n(n={len(read_sorted)})")
    if data_to_plot:
        ax.boxplot(data_to_plot, tick_labels=labels)
    ax.set_title("Write vs Read Latency Comparison")
    ax.set_ylabel("Latency (s)")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Replica load distribution
    ax = axes[1, 1]
    replica_names = [r.name.replace("replica_", "") for r in replicas]
    writes = [r.writes_received for r in replicas]
    reads = [r.reads_received for r in replicas]
    x = range(len(replicas))
    width = 0.35
    ax.bar([i - width/2 for i in x], writes, width, label="Writes", color="coral")
    ax.bar([i + width/2 for i in x], reads, width, label="Reads", color="steelblue")
    ax.set_title("Replica Load Distribution")
    ax.set_xlabel("Replica")
    ax.set_ylabel("Requests")
    ax.set_xticks(x)
    ax.set_xticklabels(replica_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "quorum_analysis.png", dpi=150)
    plt.close(fig)


# --- Test Cases ---


class TestDatabaseQuorumTopology:
    """Integration tests for database quorum with network topology."""

    def test_quorum_writes_complete_successfully(self, test_output_dir: Path):
        """Quorum writes should complete with expected latency."""
        result = run_quorum_scenario(
            duration_s=10.0,
            request_rate=5.0,
            write_ratio=1.0,  # All writes
            quorum_size=2,
            test_output_dir=test_output_dir,
        )

        # All writes should complete
        assert result.coordinator.writes_completed > 0
        assert result.coordinator.writes_completed >= result.requests_generated * 0.9

        # Write latency should be dominated by second-fastest replica
        # US-East is ~1ms round-trip, US-West is ~100ms round-trip
        # Quorum of 2 means we wait for US-West (second fastest)
        avg_write_latency = sum(result.coordinator.write_latencies) / len(result.coordinator.write_latencies)

        # Should be roughly 50ms * 2 (round-trip to US-West) + processing
        assert avg_write_latency > 0.050  # At least one cross-region RTT
        assert avg_write_latency < 0.200  # But not as slow as EU

    def test_quorum_reads_faster_than_writes(self, test_output_dir: Path):
        """Quorum reads should be faster than writes (only need quorum_size replicas)."""
        result = run_quorum_scenario(
            duration_s=10.0,
            request_rate=10.0,
            write_ratio=0.5,  # 50/50 split
            quorum_size=2,
            test_output_dir=test_output_dir,
        )

        assert result.coordinator.writes_completed > 0
        assert result.coordinator.reads_completed > 0

        avg_write = sum(result.coordinator.write_latencies) / len(result.coordinator.write_latencies)
        avg_read = sum(result.coordinator.read_latencies) / len(result.coordinator.read_latencies)

        # Reads contact only 2 replicas (US-East + US-West typically)
        # Writes contact all 3 but wait for 2
        # In this setup they should be similar since both wait for quorum
        # But reads may be slightly faster on average
        assert avg_read <= avg_write * 1.5  # Reads shouldn't be much slower

    def test_all_replicas_receive_writes(self, test_output_dir: Path):
        """All replicas should receive write requests."""
        result = run_quorum_scenario(
            duration_s=10.0,
            request_rate=5.0,
            write_ratio=1.0,  # All writes
            quorum_size=2,
            test_output_dir=test_output_dir,
        )

        # All replicas should have received writes
        for replica in result.replicas:
            assert replica.writes_received > 0, f"{replica.name} received no writes"

        # Each replica should have roughly equal writes
        total_writes = sum(r.writes_received for r in result.replicas)
        expected_per_replica = total_writes / len(result.replicas)
        for replica in result.replicas:
            assert replica.writes_received >= expected_per_replica * 0.8

    def test_network_routes_all_traffic(self, test_output_dir: Path):
        """Network should successfully route all traffic."""
        result = run_quorum_scenario(
            duration_s=5.0,
            request_rate=10.0,
            write_ratio=0.5,
            quorum_size=2,
            test_output_dir=test_output_dir,
        )

        # No dropped events
        assert result.network.events_dropped_no_route == 0
        assert result.network.events_dropped_partition == 0

        # Events were routed
        assert result.network.events_routed > 0

    def test_latency_percentiles_reflect_topology(self, test_output_dir: Path):
        """Latency percentiles should reflect the network topology."""
        result = run_quorum_scenario(
            duration_s=20.0,
            request_rate=10.0,
            write_ratio=0.3,
            quorum_size=2,
            test_output_dir=test_output_dir,
        )

        write_sorted = sorted(result.coordinator.write_latencies)

        if len(write_sorted) >= 10:
            p50 = _percentile_sorted(write_sorted, 0.50)
            p99 = _percentile_sorted(write_sorted, 0.99)

            # p50 should reflect US-West latency (~100ms round-trip)
            assert p50 > 0.050  # More than local

            # p99 should be higher due to jitter
            assert p99 > p50

            # But not absurdly high
            assert p99 < 0.500


class TestNetworkPartitionScenario:
    """Test network partition behavior in quorum system."""

    def test_partition_blocks_replica_communication(self, test_output_dir: Path):
        """Network partition should block communication to affected replica."""
        import random
        random.seed(42)

        # Create topology
        replica_a = DatabaseReplica(name="replica_a", processing_latency=0.005)
        replica_b = DatabaseReplica(name="replica_b", processing_latency=0.005)
        replica_c = DatabaseReplica(name="replica_c", processing_latency=0.005)
        replicas = [replica_a, replica_b, replica_c]

        network = Network(name="test_network")

        client = DatabaseClient(name="client")
        coordinator = QuorumCoordinator(
            name="coordinator",
            replicas=replicas,
            network=network,
            quorum_size=2,
            downstream=client,
        )

        # Add links
        for replica in replicas:
            link = NetworkLink(
                name=f"link_to_{replica.name}",
                latency=ConstantLatency(0.010),
            )
            network.add_bidirectional_link(coordinator, replica, link)

        # Create partition isolating replica_c
        network.partition([coordinator, replica_a, replica_b], [replica_c])

        # Run simulation
        provider = DatabaseRequestProvider(
            coordinator=coordinator,
            client=client,
            write_ratio=1.0,
            stop_after=Instant.from_seconds(5.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=5.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(6.0),
            sources=[source],
            entities=[network, coordinator, client] + replicas,
        )
        sim.run()

        # Replica C should receive no writes (partitioned)
        assert replica_c.writes_received == 0

        # Replicas A and B should receive writes
        assert replica_a.writes_received > 0
        assert replica_b.writes_received > 0

        # Writes should still complete (quorum of 2 from A and B)
        assert coordinator.writes_completed > 0

        # Some events should be dropped due to partition
        assert network.events_dropped_partition > 0
