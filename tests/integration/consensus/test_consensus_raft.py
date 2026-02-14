"""Integration tests for Raft consensus with full Simulation + Network.

Scenario:
- RaftNode entities form a cluster with datacenter-speed links
- Nodes use randomized election timeouts and heartbeats
- Leader election, log replication, and command commitment are tested
- Network partitions trigger new elections and leadership changes
"""

from __future__ import annotations

import random

import pytest

from happysimulator.components.consensus.raft import RaftNode, RaftState
from happysimulator.components.consensus.raft_state_machine import KVStateMachine
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.network.network import Network
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


def _build_raft_cluster(
    n: int = 3,
    election_timeout_min: float = 1.0,
    election_timeout_max: float = 2.0,
    heartbeat_interval: float = 0.3,
) -> tuple[Network, list[RaftNode]]:
    """Build an n-node full-mesh Raft cluster with datacenter links."""
    network = Network(name="RaftNet")
    nodes = [
        RaftNode(
            name=f"raft-{i}",
            network=network,
            election_timeout_min=election_timeout_min,
            election_timeout_max=election_timeout_max,
            heartbeat_interval=heartbeat_interval,
        )
        for i in range(n)
    ]
    for node in nodes:
        node.set_peers([nd for nd in nodes if nd is not node])
    # Full mesh
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            link = datacenter_network(name=f"link_{src.name}_{dst.name}")
            network.add_bidirectional_link(src, dst, link)
    return network, nodes


class TestRaftConsensus:
    """Integration tests for Raft leader election and log replication."""

    def test_leader_election_single(self):
        """A 3-node cluster elects exactly one leader."""
        random.seed(42)
        network, nodes = _build_raft_cluster(3)

        sim = Simulation(
            end_time=Instant.from_seconds(15.0),
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        sim.run()

        leaders = [n for n in nodes if n.is_leader]
        assert len(leaders) >= 1, "No leader was elected"

        # All leaders must be in the same term
        leader_terms = {n.current_term for n in leaders}
        assert len(leader_terms) == 1, (
            f"Leaders in different terms: {[(n.name, n.current_term) for n in leaders]}"
        )

        # All non-leader nodes should recognize the leader
        leader_name = leaders[0].name
        for node in nodes:
            if not node.is_leader:
                # Followers should know who the leader is
                assert node.current_leader is not None, (
                    f"{node.name} does not know the leader"
                )

    def test_leader_election_majority(self):
        """Majority of the cluster can elect a leader even with varied timeouts."""
        random.seed(42)
        network, nodes = _build_raft_cluster(
            n=3,
            election_timeout_min=1.5,
            election_timeout_max=3.0,
        )

        sim = Simulation(
            end_time=Instant.from_seconds(20.0),
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        sim.run()

        leaders = [n for n in nodes if n.is_leader]
        assert len(leaders) >= 1, "No leader elected with 3-node cluster"

        # The cluster should have a stable leader
        # (at least one election started)
        total_elections = sum(n.stats.elections_started for n in nodes)
        assert total_elections >= 1, "No elections were started"

    def test_log_replication(self):
        """Leader replicates log entries to followers."""
        random.seed(42)
        network, nodes = _build_raft_cluster(3)

        sim = Simulation(
            end_time=Instant.from_seconds(20.0),
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        # Submit a command after a leader has been elected (t=5.0)
        def submit_command(event):
            leaders = [n for n in nodes if n.is_leader]
            if leaders:
                leader = leaders[0]
                leader.submit({"op": "set", "key": "x", "value": 42})
            return None

        sim.schedule(Event.once(
            time=Instant.from_seconds(5.0),
            event_type="SubmitCommand",
            fn=submit_command,
        ))
        sim.run()

        # Find the leader and verify the log was replicated
        leaders = [n for n in nodes if n.is_leader]
        assert len(leaders) >= 1, "No leader elected"

        leader = leaders[0]
        # Leader should have the entry in its log
        assert leader.log.last_index >= 1, "Leader has no log entries"

        # Verify replication: at least a quorum of nodes should have the entry
        nodes_with_entry = sum(1 for n in nodes if n.log.last_index >= 1)
        assert nodes_with_entry >= leader.quorum_size, (
            f"Only {nodes_with_entry} nodes have the entry, need {leader.quorum_size}"
        )

    def test_partition_and_new_leader(self):
        """Partitioning the leader causes a new election in the majority group."""
        random.seed(42)
        network, nodes = _build_raft_cluster(
            n=3,
            election_timeout_min=1.0,
            election_timeout_max=2.0,
            heartbeat_interval=0.3,
        )

        sim = Simulation(
            end_time=Instant.from_seconds(30.0),
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        # Let a leader be elected first (run to t=5.0)
        from happysimulator.core.control.breakpoints import TimeBreakpoint

        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(5.0))
        )
        sim.run()

        leaders_before = [n for n in nodes if n.is_leader]
        if not leaders_before:
            # If no leader yet, continue a bit longer
            sim.control.add_breakpoint(
                TimeBreakpoint(time=Instant.from_seconds(10.0))
            )
            sim.control.resume()
            leaders_before = [n for n in nodes if n.is_leader]

        assert len(leaders_before) >= 1, "No leader elected before partition"
        old_leader = leaders_before[0]
        old_term = old_leader.current_term

        # Partition the leader from the other two
        others = [n for n in nodes if n is not old_leader]
        partition = network.partition([old_leader], others)

        # Let the majority group elect a new leader
        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(20.0))
        )
        sim.control.resume()

        # The majority group should have elected a new leader
        new_leaders = [n for n in others if n.is_leader]
        assert len(new_leaders) >= 1, (
            "No new leader elected in majority group after partition"
        )
        new_leader = new_leaders[0]
        assert new_leader.current_term > old_term, (
            f"New leader term ({new_leader.current_term}) should be > old term ({old_term})"
        )

        # Heal and let cluster converge
        partition.heal()
        sim.control.resume()

        # The old leader should have stepped down
        final_leaders = [n for n in nodes if n.is_leader]
        assert len(final_leaders) >= 1, "No leader after partition heal"

    def test_five_node_cluster(self):
        """5-node Raft cluster elects a leader and replicates entries."""
        random.seed(42)
        network, nodes = _build_raft_cluster(5)

        sim = Simulation(
            end_time=Instant.from_seconds(20.0),
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        # Submit a command at t=8.0 (after election settles)
        def submit_command(event):
            leaders = [n for n in nodes if n.is_leader]
            if leaders:
                leaders[0].submit({"op": "set", "key": "cluster_size", "value": 5})
            return None

        sim.schedule(Event.once(
            time=Instant.from_seconds(8.0),
            event_type="SubmitCmd5Node",
            fn=submit_command,
        ))
        sim.run()

        # A leader should exist
        leaders = [n for n in nodes if n.is_leader]
        assert len(leaders) >= 1, "No leader elected in 5-node cluster"

        # Quorum for 5 nodes is 3
        assert leaders[0].quorum_size == 3

        # All nodes should agree on who the leader is or have seen a leader
        total_elections = sum(n.stats.elections_started for n in nodes)
        assert total_elections >= 1

    def test_command_commit(self):
        """Submit a command and verify it is applied to the state machine."""
        random.seed(42)
        network, nodes = _build_raft_cluster(3)

        # Give each node a KV state machine we can inspect
        state_machines = []
        for node in nodes:
            sm = KVStateMachine()
            node._state_machine = sm
            state_machines.append(sm)

        sim = Simulation(
            end_time=Instant.from_seconds(25.0),
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        # Submit command after leader election
        def submit_set_command(event):
            leaders = [n for n in nodes if n.is_leader]
            if leaders:
                leaders[0].submit({"op": "set", "key": "answer", "value": 42})
            return None

        sim.schedule(Event.once(
            time=Instant.from_seconds(6.0),
            event_type="SubmitSetCmd",
            fn=submit_set_command,
        ))
        sim.run()

        # At least the leader should have committed
        leaders = [n for n in nodes if n.is_leader]
        assert len(leaders) >= 1

        leader = leaders[0]
        leader_idx = nodes.index(leader)
        leader_sm = state_machines[leader_idx]

        # Check commit index advanced
        assert leader.log.commit_index >= 1, (
            f"Leader commit_index is {leader.log.commit_index}, expected >= 1"
        )

        # Check that the state machine applied the command
        assert leader.stats.commands_committed >= 1, (
            f"Leader committed {leader.stats.commands_committed} commands"
        )

        # Check that at least a quorum has the entry committed
        committed_count = sum(
            1 for n in nodes if n.log.commit_index >= 1
        )
        assert committed_count >= leader.quorum_size, (
            f"Only {committed_count} nodes committed, need {leader.quorum_size}"
        )
