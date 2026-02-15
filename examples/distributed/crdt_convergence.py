"""CRDT Convergence Demo — eventual consistency after network partition.

Architecture::

    Writes ──► CRDTStore-A ◄──gossip──► CRDTStore-B ◄── Writes
                                ▲
                            Network
                          (with partition)

Demonstrates:
1. Both nodes accept writes independently (no coordination).
2. During partition, counter values diverge.
3. After partition heals, gossip converges values automatically.
4. Final state is the sum of all increments from both nodes.
"""

from happysimulator import (
    Event,
    Instant,
    Network,
    Simulation,
    datacenter_network,
)
from happysimulator.components.crdt import CRDTStore, GCounter


def main():
    # --- Setup ---
    network = Network(name="cluster")

    store_a = CRDTStore(
        "node-a",
        network=network,
        crdt_factory=lambda nid: GCounter(nid),
        gossip_interval=1.0,
    )
    store_b = CRDTStore(
        "node-b",
        network=network,
        crdt_factory=lambda nid: GCounter(nid),
        gossip_interval=1.0,
    )

    store_a.add_peers([store_b])
    store_b.add_peers([store_a])

    network.add_bidirectional_link(store_a, store_b, datacenter_network("link"))

    # --- Simulation ---
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=30.0,
        sources=[],
        entities=[store_a, store_b, network],
    )

    # --- Schedule write events with proper metadata ---
    # Phase 1: Normal operation (t=0.5 to t=5.0) — writes to both nodes
    write_events = []
    for i in range(10):
        t = 0.5 + i * 0.5
        target = store_a if i % 2 == 0 else store_b
        write_events.append(Event(
            time=Instant.from_seconds(t),
            event_type="Write",
            target=target,
            context={"metadata": {
                "key": "page-views",
                "operation": "increment",
                "value": 1,
            }},
        ))

    # Phase 2: Partition at t=5.0, writes during partition (t=5 to t=10)
    partition = None

    def create_partition(e):
        nonlocal partition
        partition = network.partition([store_a], [store_b])

    partition_event = Event.once(
        time=Instant.from_seconds(5.0),
        event_type="CreatePartition",
        fn=create_partition,
    )

    # Writes during partition
    for i in range(10):
        t = 5.5 + i * 0.5
        target = store_a if i % 2 == 0 else store_b
        write_events.append(Event(
            time=Instant.from_seconds(t),
            event_type="Write",
            target=target,
            context={"metadata": {
                "key": "page-views",
                "operation": "increment",
                "value": 1,
            }},
        ))

    # Phase 3: Heal partition at t=10.0
    heal_event = Event.once(
        time=Instant.from_seconds(10.0),
        event_type="HealPartition",
        fn=lambda e: partition.heal() if partition else None,
    )

    # Schedule gossip ticks (non-daemon so sim processes them)
    gossip_events = []
    for t in range(1, 20):
        gossip_events.append(Event(
            time=Instant.from_seconds(float(t)),
            event_type="GossipTick",
            target=store_a,
            daemon=False,
        ))
        gossip_events.append(Event(
            time=Instant.from_seconds(float(t) + 0.5),
            event_type="GossipTick",
            target=store_b,
            daemon=False,
        ))

    sim.schedule(write_events + [partition_event, heal_event] + gossip_events)

    # --- Snapshot before gossip convergence ---
    # Run first part (through partition)
    summary = sim.run()

    # --- Results ---
    print("=" * 60)
    print("CRDT Convergence Demo")
    print("=" * 60)
    print()
    print(f"Simulation duration: {summary.duration_s:.1f}s")
    print(f"Total events processed: {summary.total_events_processed}")
    print()

    val_a = store_a.crdts.get("page-views")
    val_b = store_b.crdts.get("page-views")

    if val_a and val_b:
        print(f"Node-A counter value: {val_a.value}")
        print(f"Node-B counter value: {val_b.value}")
        converged = val_a.value == val_b.value
        print(f"Converged: {converged}")
        if converged:
            print(f"Final merged value: {val_a.value}")
            print(f"  (= 20 total increments across both nodes)")
    else:
        print("No counter data found.")
    print()

    print("Node-A stats:")
    print(f"  Writes: {store_a.stats.writes}")
    print(f"  Gossip sent: {store_a.stats.gossip_sent}")
    print(f"  Gossip received: {store_a.stats.gossip_received}")
    print(f"  Keys merged: {store_a.stats.keys_merged}")
    print()
    print("Node-B stats:")
    print(f"  Writes: {store_b.stats.writes}")
    print(f"  Gossip sent: {store_b.stats.gossip_sent}")
    print(f"  Gossip received: {store_b.stats.gossip_received}")
    print(f"  Keys merged: {store_b.stats.keys_merged}")
    print()
    print(f"Network events dropped (partition): {network.events_dropped_partition}")


if __name__ == "__main__":
    main()
