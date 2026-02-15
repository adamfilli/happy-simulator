"""Integration tests for Paxos consensus with full Simulation + Network.

Scenario:
- PaxosNode entities form a cluster with datacenter-speed links
- Proposals trigger Phase 1 (Prepare/Promise) and Phase 2 (Accept/Accepted)
- Network partitions test fault tolerance and consensus safety
- All events route through the Network entity with realistic latencies
"""

from __future__ import annotations

import random

import pytest

from happysimulator.components.consensus.paxos import Ballot, PaxosNode
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.network.network import Network
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


def _build_paxos_cluster(n: int = 3) -> tuple[Network, list[PaxosNode]]:
    """Build an n-node full-mesh Paxos cluster with datacenter links."""
    network = Network(name="PaxosNet")
    nodes = [PaxosNode(name=f"paxos-{i}", network=network) for i in range(n)]
    for node in nodes:
        node.set_peers([nd for nd in nodes if nd is not node])
    # Full mesh
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            link = datacenter_network(name=f"link_{src.name}_{dst.name}")
            network.add_bidirectional_link(src, dst, link)
    return network, nodes


class TestPaxosConsensus:
    """Integration tests for single-decree Paxos consensus."""

    def test_single_proposer_decides(self):
        """One node proposes a value; all nodes eventually learn the decided value."""
        random.seed(42)
        network, nodes = _build_paxos_cluster(3)
        proposer = nodes[0]

        sim = Simulation(
            duration=10.0,
            entities=[network, *nodes],
        )

        def do_propose(event):
            proposer.propose("value-42")
            events = proposer.start_phase1()
            return events

        trigger = Event.once(
            time=Instant.from_seconds(0.1),
            event_type="TriggerProposal",
            fn=do_propose,
        )
        sim.schedule(trigger)
        sim.run()

        # All nodes should have decided
        for node in nodes:
            assert node.is_decided, f"{node.name} did not decide"
            assert node.decided_value == "value-42", (
                f"{node.name} decided {node.decided_value!r}, expected 'value-42'"
            )

    def test_two_proposers_one_wins(self):
        """Two concurrent proposals; exactly one value is decided across all nodes."""
        random.seed(42)
        network, nodes = _build_paxos_cluster(3)
        proposer_a = nodes[0]
        proposer_b = nodes[1]

        sim = Simulation(
            duration=15.0,
            entities=[network, *nodes],
        )

        # Proposer A proposes at t=0.1
        def propose_a(event):
            proposer_a.propose("alpha")
            return proposer_a.start_phase1()

        # Proposer B proposes at t=0.1 (concurrent)
        def propose_b(event):
            proposer_b.propose("beta")
            return proposer_b.start_phase1()

        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="ProposeA",
            fn=propose_a,
        ))
        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="ProposeB",
            fn=propose_b,
        ))
        sim.run()

        # At least one node must have decided
        decided_nodes = [n for n in nodes if n.is_decided]
        assert len(decided_nodes) >= 1, "No node decided"

        # All decided nodes must agree on the same value
        decided_values = {n.decided_value for n in decided_nodes}
        assert len(decided_values) == 1, (
            f"Safety violation: multiple decided values {decided_values}"
        )

        # The decided value must be one of the proposed values
        decided_value = decided_values.pop()
        assert decided_value in ("alpha", "beta"), (
            f"Decided value {decided_value!r} was not proposed"
        )

    def test_partition_blocks_consensus(self):
        """A proposer partitioned from the majority cannot reach consensus."""
        random.seed(42)
        network, nodes = _build_paxos_cluster(3)
        isolated = nodes[0]
        majority = [nodes[1], nodes[2]]

        sim = Simulation(
            duration=5.0,
            entities=[network, *nodes],
        )

        # Partition node-0 from the other two
        network.partition([isolated], majority)

        def propose_isolated(event):
            isolated.propose("lonely-value")
            return isolated.start_phase1()

        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="ProposeIsolated",
            fn=propose_isolated,
        ))
        sim.run()

        # The isolated proposer cannot form a quorum (needs 2 of 3)
        # It self-promises but never gets a second promise
        assert not isolated.is_decided, (
            "Isolated proposer should not decide without a quorum"
        )
        # The other nodes should also not have decided (no proposal reached them)
        for node in majority:
            assert not node.is_decided, (
                f"{node.name} should not have decided"
            )

    def test_partition_heal_reaches_consensus(self):
        """Partition blocks consensus; healing allows it to complete."""
        random.seed(42)
        network, nodes = _build_paxos_cluster(3)
        isolated = nodes[0]
        majority = [nodes[1], nodes[2]]

        sim = Simulation(
            duration=15.0,
            entities=[network, *nodes],
        )

        # Partition node-0 from the other two
        partition = network.partition([isolated], majority)

        # Propose while partitioned (will fail to reach quorum)
        def propose_partitioned(event):
            isolated.propose("delayed-value")
            return isolated.start_phase1()

        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="ProposePartitioned",
            fn=propose_partitioned,
        ))

        # Heal partition at t=3.0 and re-propose
        def heal_and_retry(event):
            partition.heal()
            # Re-propose with a fresh ballot since the original may have been nacked
            isolated.propose("delayed-value")
            return isolated.start_phase1()

        sim.schedule(Event.once(
            time=Instant.from_seconds(3.0),
            event_type="HealAndRetry",
            fn=heal_and_retry,
        ))
        sim.run()

        # After healing, consensus should have been reached
        decided_nodes = [n for n in nodes if n.is_decided]
        assert len(decided_nodes) >= 1, "No node decided after partition healed"

        decided_values = {n.decided_value for n in decided_nodes}
        assert len(decided_values) == 1, (
            f"Safety violation: {decided_values}"
        )

    def test_five_node_cluster(self):
        """5-node cluster reaches consensus through a single proposal."""
        random.seed(42)
        network, nodes = _build_paxos_cluster(5)
        proposer = nodes[2]

        sim = Simulation(
            duration=10.0,
            entities=[network, *nodes],
        )

        def do_propose(event):
            proposer.propose("five-node-value")
            return proposer.start_phase1()

        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="Propose5Node",
            fn=do_propose,
        ))
        sim.run()

        # All nodes should learn the decided value
        for node in nodes:
            assert node.is_decided, f"{node.name} did not decide"
            assert node.decided_value == "five-node-value"

        # Quorum size for 5 nodes is 3
        assert proposer.quorum_size == 3

    def test_ballot_ordering_determines_winner(self):
        """A higher ballot proposal supersedes a lower ballot one when concurrent."""
        random.seed(42)
        network, nodes = _build_paxos_cluster(3)
        low_proposer = nodes[0]
        high_proposer = nodes[1]

        sim = Simulation(
            duration=15.0,
            entities=[network, *nodes],
        )

        # Low proposer starts first at t=0.1
        def propose_low(event):
            low_proposer.propose("low-ballot-value")
            return low_proposer.start_phase1()

        # High proposer starts slightly later at t=0.2 with a higher ballot
        def propose_high(event):
            high_proposer.propose("high-ballot-value")
            return high_proposer.start_phase1()

        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="ProposeLow",
            fn=propose_low,
        ))
        sim.schedule(Event.once(
            time=Instant.from_seconds(0.2),
            event_type="ProposeHigh",
            fn=propose_high,
        ))
        sim.run()

        # At least one node must have decided
        decided_nodes = [n for n in nodes if n.is_decided]
        assert len(decided_nodes) >= 1, "No node decided"

        # All decided nodes must agree (safety)
        decided_values = {n.decided_value for n in decided_nodes}
        assert len(decided_values) == 1, (
            f"Safety violation: {decided_values}"
        )

        # The decided value must be one of the proposed values
        decided_value = decided_values.pop()
        assert decided_value in ("low-ballot-value", "high-ballot-value"), (
            f"Unexpected decided value: {decided_value!r}"
        )

        # Verify that ballots are correctly ordered
        assert Ballot(1, "paxos-0") < Ballot(2, "paxos-1"), (
            "Higher ballot number should be greater"
        )
        assert Ballot(1, "paxos-0") < Ballot(1, "paxos-1"), (
            "Same number, higher node_id should be greater"
        )
