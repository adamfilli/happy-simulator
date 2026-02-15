"""Single-decree Paxos consensus reaching agreement on a value.

This example demonstrates the classic Paxos protocol:
1. A 3-node cluster is created with datacenter network links
2. One node proposes a value
3. The protocol executes Phase 1 (Prepare/Promise) and Phase 2 (Accept/Accepted)
4. All nodes learn the decided value
5. Statistics show ballot progression and message counts

## Architecture Diagram

```
+------------------------------------------------------------------+
|                    PAXOS CONSENSUS SIMULATION                     |
+------------------------------------------------------------------+

    +----------+       datacenter links       +----------+
    |  Node 1  |<---------------------------->|  Node 2  |
    | (proposer)|                              |          |
    +----+-----+                              +----+-----+
         |                                         |
         |              +----------+               |
         +------------->|  Node 3  |<--------------+
                        |          |
                        +----------+

    Phase 1: Node 1 sends Prepare(ballot=1) to all
             Nodes 2,3 respond with Promise
    Phase 2: Node 1 sends Accept(value="alpha") to all
             Nodes 2,3 respond with Accepted
    Decision: All nodes learn decided value "alpha"
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.core.event import Event
from happysimulator.components.network.network import Network
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.consensus.paxos import PaxosNode, PaxosStats


# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the Paxos consensus simulation."""
    nodes: list[PaxosNode]
    proposer: PaxosNode
    proposed_value: str
    network: Network
    duration_s: float


# =============================================================================
# Main Simulation
# =============================================================================


def run(args=None) -> SimulationResult:
    """Run a single-decree Paxos simulation with 3 nodes.

    Args:
        args: Optional argparse namespace with simulation parameters.

    Returns:
        SimulationResult with all node states and statistics.
    """
    duration_s = getattr(args, "duration", 5.0) if args else 5.0
    seed = getattr(args, "seed", 42) if args else 42

    if seed is not None and seed >= 0:
        random.seed(seed)

    # Create network
    network = Network(name="paxos-net")

    # Create 3 Paxos nodes
    node1 = PaxosNode(name="node-1", network=network, retry_delay=0.5)
    node2 = PaxosNode(name="node-2", network=network, retry_delay=0.5)
    node3 = PaxosNode(name="node-3", network=network, retry_delay=0.5)

    nodes = [node1, node2, node3]

    # Wire up peers
    for node in nodes:
        node.set_peers(nodes)

    # Create datacenter links between all pairs
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            network.add_bidirectional_link(a, b, datacenter_network(f"link-{a.name}-{b.name}"))

    # Propose a value from node1 using Event.once() trigger
    proposed_value = "alpha"

    def trigger_proposal(event: Event):
        node1.propose(proposed_value)
        return node1.start_phase1()

    trigger = Event.once(
        time=Instant.from_seconds(0.1),
        event_type="TriggerProposal",
        fn=trigger_proposal,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s,
        entities=[network, node1, node2, node3],
    )
    sim.schedule(trigger)
    sim.run()

    return SimulationResult(
        nodes=nodes,
        proposer=node1,
        proposed_value=proposed_value,
        network=network,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary of the Paxos consensus result."""
    print("\n" + "=" * 60)
    print("PAXOS CONSENSUS SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nCluster size: {len(result.nodes)} nodes")
    print(f"Quorum size:  {result.nodes[0].quorum_size}")
    print(f"Proposed value: {result.proposed_value!r}")
    print(f"Proposer: {result.proposer.name}")

    print(f"\nNode States:")
    print(f"  {'Node':<10} {'Decided':<10} {'Value':<15} {'Proposals':<12} "
          f"{'Promises':<10} {'Accepts':<10} {'Nacks':<8}")
    print(f"  {'-' * 75}")

    all_decided = True
    for node in result.nodes:
        s = node.stats
        decided_str = "YES" if node.is_decided else "NO"
        value_str = repr(s.decided_value) if s.decided_value is not None else "-"
        print(f"  {node.name:<10} {decided_str:<10} {value_str:<15} "
              f"{s.proposals_started:<12} {s.promises_received:<10} "
              f"{s.accepts_received:<10} {s.nacks_received:<8}")
        if not node.is_decided:
            all_decided = False

    print(f"\nConsensus reached: {'YES' if all_decided else 'NO'}")

    # Check all decided on same value
    decided_values = {node.decided_value for node in result.nodes if node.is_decided}
    if len(decided_values) == 1:
        print(f"Agreed value: {decided_values.pop()!r}")
    elif len(decided_values) > 1:
        print(f"SAFETY VIOLATION: Multiple decided values: {decided_values}")

    print(f"\nNetwork Statistics:")
    print(f"  Events routed:  {result.network.events_routed}")
    print(f"  Dropped (partition): {result.network.events_dropped_partition}")
    print(f"  Dropped (no route):  {result.network.events_dropped_no_route}")

    print("\n" + "=" * 60)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualization of Paxos ballot and message statistics."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: message counts per node
    ax = axes[0]
    node_names = [n.name for n in result.nodes]
    promises = [n.stats.promises_received for n in result.nodes]
    accepts = [n.stats.accepts_received for n in result.nodes]
    nacks = [n.stats.nacks_received for n in result.nodes]

    x = range(len(node_names))
    width = 0.25
    ax.bar([i - width for i in x], promises, width, label="Promises", color="steelblue")
    ax.bar(x, accepts, width, label="Accepts", color="seagreen")
    ax.bar([i + width for i in x], nacks, width, label="Nacks", color="indianred")
    ax.set_xlabel("Node")
    ax.set_ylabel("Message Count")
    ax.set_title("Paxos Messages per Node")
    ax.set_xticks(x)
    ax.set_xticklabels(node_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Pie chart: proposal outcome
    ax = axes[1]
    proposer_stats = result.proposer.stats
    succeeded = proposer_stats.proposals_succeeded
    failed = proposer_stats.proposals_failed
    pending = proposer_stats.proposals_started - succeeded - failed
    sizes = [succeeded, failed, pending]
    labels = [f"Succeeded ({succeeded})", f"Failed ({failed})", f"Pending ({pending})"]
    colors = ["seagreen", "indianred", "gold"]
    # Filter out zero slices
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
    ax.set_title("Proposal Outcomes")

    fig.suptitle("Single-Decree Paxos Consensus", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "paxos_consensus.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'paxos_consensus.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single-decree Paxos consensus simulation")
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation duration (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/paxos", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = None

    print("Running single-decree Paxos consensus simulation...")
    print(f"  Duration: {args.duration}s")
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
