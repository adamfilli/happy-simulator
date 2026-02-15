"""Integration tests for SWIM-style membership protocol with full Simulation + Network.

Scenario:
- MembershipProtocol entities form a cluster with datacenter-speed links
- Nodes periodically probe peers using ping/ack with phi accrual detection
- Network partitions cause suspicion and eventually death declarations
- Healing partitions allows recovery of membership state
"""

from __future__ import annotations

import random

import pytest

from happysimulator.components.consensus.membership import (
    MembershipProtocol,
    MemberState,
)
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.network.network import Network
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


def _build_membership_cluster(
    n: int = 5,
    probe_interval: float = 0.5,
    suspicion_timeout: float = 3.0,
    phi_threshold: float = 8.0,
) -> tuple[Network, list[MembershipProtocol]]:
    """Build an n-node full-mesh membership cluster with datacenter links."""
    network = Network(name="MemberNet")
    nodes = [
        MembershipProtocol(
            name=f"member-{i}",
            network=network,
            probe_interval=probe_interval,
            suspicion_timeout=suspicion_timeout,
            phi_threshold=phi_threshold,
        )
        for i in range(n)
    ]
    # Each node adds all others as members
    for node in nodes:
        for other in nodes:
            if other is not node:
                node.add_member(other)
    # Full mesh network links
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            link = datacenter_network(name=f"link_{src.name}_{dst.name}")
            network.add_bidirectional_link(src, dst, link)
    return network, nodes


class TestMembershipProtocol:
    """Integration tests for SWIM membership failure detection."""

    def test_healthy_cluster_all_alive(self):
        """In a healthy cluster, all members remain ALIVE throughout the run."""
        random.seed(42)
        network, nodes = _build_membership_cluster(
            n=5,
            probe_interval=0.5,
            suspicion_timeout=5.0,
        )

        sim = Simulation(
            duration=20.0,
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        sim.run()

        # Every node should see all other members as ALIVE
        for node in nodes:
            alive = node.alive_members
            suspected = node.suspected_members
            dead = node.dead_members

            # All other members should be alive (no false positives)
            assert len(alive) == len(nodes) - 1, (
                f"{node.name}: expected {len(nodes) - 1} alive members, "
                f"got {len(alive)} (suspected={suspected}, dead={dead})"
            )
            assert len(dead) == 0, (
                f"{node.name}: unexpected dead members: {dead}"
            )

        # All nodes should have sent probes
        for node in nodes:
            assert node.stats.probes_sent > 0, (
                f"{node.name} sent no probes"
            )
            assert node.stats.acks_received > 0, (
                f"{node.name} received no acks"
            )

    def test_stopped_node_detected(self):
        """A node that stops responding is eventually detected as SUSPECT or DEAD."""
        random.seed(42)
        network, nodes = _build_membership_cluster(
            n=5,
            probe_interval=0.5,
            suspicion_timeout=3.0,
            phi_threshold=4.0,  # Lower threshold for faster detection
        )

        sim = Simulation(
            duration=30.0,
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        # Let the cluster stabilize first (run to t=5.0)
        from happysimulator.core.control.breakpoints import TimeBreakpoint

        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(5.0))
        )
        sim.run()

        # Partition member-4 from ALL other nodes (simulates a stopped node)
        stopped = nodes[4]
        active_nodes = nodes[:4]
        partition = network.partition([stopped], active_nodes)

        # Continue running to let failure detection kick in
        sim.control.resume()

        # At least some active nodes should detect member-4 as suspect or dead
        detection_count = 0
        for node in active_nodes:
            state = node.get_member_state(stopped.name)
            if state in (MemberState.SUSPECT, MemberState.DEAD):
                detection_count += 1

        assert detection_count >= 1, (
            "No active nodes detected the stopped node as suspect/dead. "
            f"States: {[(n.name, n.get_member_state(stopped.name)) for n in active_nodes]}"
        )

    def test_partition_causes_suspicion(self):
        """A network partition between groups causes cross-partition suspicion."""
        random.seed(42)
        network, nodes = _build_membership_cluster(
            n=5,
            probe_interval=0.5,
            suspicion_timeout=3.0,
            phi_threshold=4.0,
        )

        sim = Simulation(
            duration=25.0,
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        # Let the cluster warm up
        from happysimulator.core.control.breakpoints import TimeBreakpoint

        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(3.0))
        )
        sim.run()

        # Partition {member-0, member-1} from {member-2, member-3, member-4}
        group_a = [nodes[0], nodes[1]]
        group_b = [nodes[2], nodes[3], nodes[4]]
        network.partition(group_a, group_b)

        # Run long enough for suspicion to develop
        sim.control.resume()

        # Nodes in group_a should suspect at least some nodes from group_b
        for node in group_a:
            cross_partition_suspected = 0
            for other_node in group_b:
                state = node.get_member_state(other_node.name)
                if state in (MemberState.SUSPECT, MemberState.DEAD):
                    cross_partition_suspected += 1
            assert cross_partition_suspected >= 1, (
                f"{node.name} did not suspect any cross-partition members. "
                f"States: {[(n.name, node.get_member_state(n.name)) for n in group_b]}"
            )

        # Nodes in group_b should suspect at least some nodes from group_a
        for node in group_b:
            cross_partition_suspected = 0
            for other_node in group_a:
                state = node.get_member_state(other_node.name)
                if state in (MemberState.SUSPECT, MemberState.DEAD):
                    cross_partition_suspected += 1
            assert cross_partition_suspected >= 1, (
                f"{node.name} did not suspect any cross-partition members. "
                f"States: {[(n.name, node.get_member_state(n.name)) for n in group_a]}"
            )

    def test_partition_heal_recovers(self):
        """Healing a partition allows nodes to recover to ALIVE state."""
        random.seed(42)
        network, nodes = _build_membership_cluster(
            n=5,
            probe_interval=0.5,
            suspicion_timeout=3.0,
            phi_threshold=4.0,
        )

        sim = Simulation(
            duration=40.0,
            entities=[network, *nodes],
        )
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        from happysimulator.core.control.breakpoints import TimeBreakpoint

        # Phase 1: Warm up (0-3s)
        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(3.0))
        )
        sim.run()

        # Phase 2: Create partition (3-15s)
        group_a = [nodes[0], nodes[1]]
        group_b = [nodes[2], nodes[3], nodes[4]]
        partition = network.partition(group_a, group_b)

        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(15.0))
        )
        sim.control.resume()

        # Verify some suspicion/death developed
        suspicions_before_heal = 0
        for node in group_a:
            for other in group_b:
                state = node.get_member_state(other.name)
                if state in (MemberState.SUSPECT, MemberState.DEAD):
                    suspicions_before_heal += 1
        assert suspicions_before_heal >= 1, (
            "Expected at least some cross-partition suspicion before healing"
        )

        # Phase 3: Heal and let recovery happen (15-40s)
        partition.heal()
        sim.control.resume()

        # After healing: nodes that were only SUSPECT (not DEAD) should recover
        # For nodes declared DEAD, recovery depends on re-announcement (which
        # this protocol doesn't implement). But nodes only suspected should be
        # alive again after receiving acks.
        alive_count_in_a = 0
        for node in group_a:
            for other in group_b:
                state = node.get_member_state(other.name)
                if state == MemberState.ALIVE:
                    alive_count_in_a += 1

        alive_count_in_b = 0
        for node in group_b:
            for other in group_a:
                state = node.get_member_state(other.name)
                if state == MemberState.ALIVE:
                    alive_count_in_b += 1

        total_recovered = alive_count_in_a + alive_count_in_b
        # At least some cross-partition members should have recovered
        # (those not yet declared DEAD before healing)
        assert total_recovered >= 1, (
            "No cross-partition members recovered after healing. "
            f"group_a sees group_b: "
            f"{[(n.name, [(o.name, n.get_member_state(o.name)) for o in group_b]) for n in group_a]}, "
            f"group_b sees group_a: "
            f"{[(n.name, [(o.name, n.get_member_state(o.name)) for o in group_a]) for n in group_b]}"
        )
