"""Raft leader election and log replication demonstration.

This example demonstrates the Raft consensus protocol:
1. A 5-node cluster starts with all nodes as followers
2. Election timeouts trigger leader election
3. A leader is elected and begins sending heartbeats
4. Commands are submitted and replicated to followers
5. Committed entries are applied to the KV state machine

## Architecture Diagram

```
+------------------------------------------------------------------+
|                  RAFT LEADER ELECTION SIMULATION                  |
+------------------------------------------------------------------+

    +---------+     +---------+     +---------+
    | Node 1  |<--->| Node 2  |<--->| Node 3  |
    |Follower |     |Follower |     |Follower |
    +----+----+     +----+----+     +----+----+
         |               |               |
         |          +----+----+          |
         +--------->| Node 4  |<---------+
         |          |Follower |          |
         |          +----+----+          |
         |               |               |
         |          +----+----+          |
         +--------->| Node 5  |<---------+
                    |Follower |
                    +---------+

    t=0:   All nodes start as FOLLOWER
    t~1.5: First election timeout fires -> CANDIDATE
    t~2:   Leader elected -> LEADER sends heartbeats
    t=3:   Submit commands to leader
    t=5:   Commands replicated and committed
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.core.event import Event
from happysimulator.components.network.network import Network
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.consensus.raft import RaftNode, RaftStats, RaftState
from happysimulator.components.consensus.raft_state_machine import KVStateMachine


# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the Raft leader election simulation."""
    nodes: list[RaftNode]
    state_machines: dict[str, KVStateMachine]
    commands_submitted: list[dict[str, Any]]
    network: Network
    duration_s: float


# =============================================================================
# Main Simulation
# =============================================================================


def run(args=None) -> SimulationResult:
    """Run a Raft leader election and replication simulation.

    Args:
        args: Optional argparse namespace with simulation parameters.

    Returns:
        SimulationResult with all node states and statistics.
    """
    duration_s = getattr(args, "duration", 10.0) if args else 10.0
    num_nodes = getattr(args, "nodes", 5) if args else 5
    seed = getattr(args, "seed", 42) if args else 42

    if seed is not None and seed >= 0:
        random.seed(seed)

    # Create network
    network = Network(name="raft-net")

    # Create state machines (one per node, for inspection)
    state_machines: dict[str, KVStateMachine] = {}

    # Create Raft nodes
    nodes: list[RaftNode] = []
    for i in range(1, num_nodes + 1):
        name = f"node-{i}"
        sm = KVStateMachine()
        state_machines[name] = sm
        node = RaftNode(
            name=name,
            network=network,
            state_machine=sm,
            election_timeout_min=1.5,
            election_timeout_max=3.0,
            heartbeat_interval=0.5,
        )
        nodes.append(node)

    # Wire up peers
    for node in nodes:
        node.set_peers(nodes)

    # Create datacenter links between all pairs
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            network.add_bidirectional_link(
                a, b, datacenter_network(f"link-{a.name}-{b.name}")
            )

    # Schedule node start events (each node starts its election timeout)
    start_events: list[Event] = []
    for node in nodes:

        def make_start_fn(n: RaftNode):
            def start_fn(event: Event):
                return n.start()
            return start_fn

        evt = Event.once(
            time=Instant.from_seconds(0.01),
            event_type="StartNode",
            fn=make_start_fn(node),
        )
        start_events.append(evt)

    # Schedule commands to submit after leader election has had time to complete
    commands_submitted: list[dict[str, Any]] = [
        {"op": "set", "key": "x", "value": 1},
        {"op": "set", "key": "y", "value": 2},
        {"op": "set", "key": "z", "value": 3},
    ]

    def submit_commands(event: Event):
        """Find the leader and submit commands."""
        leader = None
        for n in nodes:
            if n.is_leader:
                leader = n
                break

        if leader is None:
            # No leader yet; try again later
            retry = Event.once(
                time=event.time + 1.0,
                event_type="RetrySubmit",
                fn=submit_commands,
            )
            return [retry]

        events = []
        for i, cmd in enumerate(commands_submitted):
            def make_submit_fn(ld: RaftNode, command: dict):
                def fn(e: Event):
                    ld.submit(command)
                    return None
                return fn

            submit_evt = Event.once(
                time=event.time + 0.1 * i,
                event_type="SubmitCommand",
                fn=make_submit_fn(leader, cmd),
            )
            events.append(submit_evt)
        return events

    submit_trigger = Event.once(
        time=Instant.from_seconds(3.0),
        event_type="SubmitCommands",
        fn=submit_commands,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s),
        entities=[network] + nodes,
    )

    for evt in start_events:
        sim.schedule(evt)
    sim.schedule(submit_trigger)

    sim.run()

    return SimulationResult(
        nodes=nodes,
        state_machines=state_machines,
        commands_submitted=commands_submitted,
        network=network,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary of the Raft consensus result."""
    print("\n" + "=" * 70)
    print("RAFT LEADER ELECTION SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nCluster size: {len(result.nodes)} nodes")
    print(f"Quorum size:  {result.nodes[0].quorum_size}")
    print(f"Commands submitted: {len(result.commands_submitted)}")

    # Find the leader
    leaders = [n for n in result.nodes if n.is_leader]
    if leaders:
        leader = leaders[0]
        print(f"\nLeader: {leader.name} (term {leader.current_term})")
    else:
        print("\nNo leader elected (cluster may be partitioned)")

    print(f"\nNode States:")
    print(f"  {'Node':<10} {'State':<12} {'Term':<6} {'Leader':<10} "
          f"{'Log Len':<9} {'Committed':<11} {'Elections':<11} {'Votes Rcvd':<10}")
    print(f"  {'-' * 89}")

    for node in result.nodes:
        s = node.stats
        state_str = s.state.name
        leader_str = s.current_leader or "-"
        print(f"  {node.name:<10} {state_str:<12} {s.current_term:<6} "
              f"{leader_str:<10} {s.log_length:<9} {s.commit_index:<11} "
              f"{s.elections_started:<11} {s.votes_received:<10}")

    # Show state machine contents
    print(f"\nState Machine Contents (leader):")
    if leaders:
        sm = result.state_machines[leaders[0].name]
        data = sm.data
        if data:
            for key, value in sorted(data.items()):
                print(f"  {key} = {value}")
        else:
            print("  (empty)")

    # Check replication consistency
    print(f"\nReplication Consistency:")
    sm_snapshots = {}
    for name, sm in result.state_machines.items():
        sm_snapshots[name] = sm.data
    unique_states = set(str(sorted(v.items())) for v in sm_snapshots.values())
    if len(unique_states) == 1:
        print("  All nodes have consistent state machine state")
    else:
        print("  WARNING: State machines are not yet consistent")
        for name, snap in sm_snapshots.items():
            print(f"    {name}: {snap}")

    print(f"\nNetwork Statistics:")
    print(f"  Events routed:  {result.network.events_routed}")
    print(f"  Dropped (partition): {result.network.events_dropped_partition}")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualization of Raft state transitions and log replication."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    node_names = [n.name for n in result.nodes]

    # Bar chart: log length vs commit index per node
    ax = axes[0]
    log_lengths = [n.stats.log_length for n in result.nodes]
    commit_indices = [n.stats.commit_index for n in result.nodes]

    x = range(len(node_names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], log_lengths, width, label="Log Length", color="steelblue")
    ax.bar([i + width / 2 for i in x], commit_indices, width, label="Commit Index", color="seagreen")
    ax.set_xlabel("Node")
    ax.set_ylabel("Log Index")
    ax.set_title("Log Replication State")
    ax.set_xticks(x)
    ax.set_xticklabels(node_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Node state summary (horizontal bar showing state)
    ax = axes[1]
    state_colors = {
        RaftState.FOLLOWER: "steelblue",
        RaftState.CANDIDATE: "gold",
        RaftState.LEADER: "seagreen",
    }
    states = [n.stats.state for n in result.nodes]
    colors = [state_colors[s] for s in states]
    terms = [n.stats.current_term for n in result.nodes]

    bars = ax.barh(node_names, terms, color=colors)
    ax.set_xlabel("Current Term")
    ax.set_title("Node States and Terms")

    # Add state labels on bars
    for bar, state in zip(bars, states):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                state.name, va="center", fontsize=10)

    ax.grid(True, alpha=0.3, axis="x")

    # Legend for states
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Follower"),
        Patch(facecolor="gold", label="Candidate"),
        Patch(facecolor="seagreen", label="Leader"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.suptitle("Raft Leader Election & Log Replication", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "raft_leader_election.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'raft_leader_election.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Raft leader election simulation")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--nodes", type=int, default=5, help="Number of Raft nodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/raft", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = None

    print("Running Raft leader election simulation...")
    print(f"  Duration: {args.duration}s")
    print(f"  Nodes: {args.nodes}")
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
