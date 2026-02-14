"""Unit tests for behavior social_network module."""

import random

from happysimulator.components.behavior.social_network import (
    Relationship,
    SocialGraph,
)


class TestRelationship:
    def test_creation(self):
        r = Relationship(source="a", target="b", weight=0.8, trust=0.6)
        assert r.source == "a"
        assert r.target == "b"
        assert r.weight == 0.8
        assert r.trust == 0.6
        assert r.interaction_count == 0


class TestSocialGraph:
    def test_add_node(self):
        g = SocialGraph()
        g.add_node("alice")
        assert "alice" in g.nodes

    def test_add_edge(self):
        g = SocialGraph()
        rel = g.add_edge("alice", "bob", weight=0.9)
        assert rel.source == "alice"
        assert rel.target == "bob"
        assert "bob" in g.neighbors("alice")
        assert "alice" not in g.neighbors("bob")  # directed

    def test_add_bidirectional_edge(self):
        g = SocialGraph()
        r1, r2 = g.add_bidirectional_edge("alice", "bob")
        assert "bob" in g.neighbors("alice")
        assert "alice" in g.neighbors("bob")

    def test_get_edge(self):
        g = SocialGraph()
        g.add_edge("a", "b", weight=0.3)
        assert g.get_edge("a", "b") is not None
        assert g.get_edge("a", "b").weight == 0.3
        assert g.get_edge("b", "a") is None

    def test_influencers(self):
        g = SocialGraph()
        g.add_edge("a", "c")
        g.add_edge("b", "c")
        influencers = g.influencers("c")
        assert set(influencers) == {"a", "b"}

    def test_influence_weights(self):
        g = SocialGraph()
        g.add_edge("a", "c", weight=0.3)
        g.add_edge("b", "c", weight=0.7)
        weights = g.influence_weights("c")
        assert weights == {"a": 0.3, "b": 0.7}

    def test_record_interaction(self):
        g = SocialGraph()
        g.add_edge("a", "b")
        g.record_interaction("a", "b")
        g.record_interaction("a", "b")
        assert g.get_edge("a", "b").interaction_count == 2

    def test_edge_count(self):
        g = SocialGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        g.add_edge("a", "c")
        assert g.edge_count == 3


class TestSocialGraphGenerators:
    def test_complete(self):
        names = ["a", "b", "c"]
        g = SocialGraph.complete(names)
        # 3 nodes, each connects to 2 others (bidirectional)
        for name in names:
            assert len(g.neighbors(name)) == 2

    def test_random_erdos_renyi(self):
        names = [f"n{i}" for i in range(20)]
        g = SocialGraph.random_erdos_renyi(names, p=0.5, rng=random.Random(42))
        assert g.edge_count > 0
        assert g.nodes == set(names)

    def test_random_deterministic(self):
        names = ["a", "b", "c", "d"]
        g1 = SocialGraph.random_erdos_renyi(names, p=0.5, rng=random.Random(42))
        g2 = SocialGraph.random_erdos_renyi(names, p=0.5, rng=random.Random(42))
        assert g1.edge_count == g2.edge_count

    def test_small_world(self):
        names = [f"n{i}" for i in range(20)]
        g = SocialGraph.small_world(names, k=4, p_rewire=0.1, rng=random.Random(42))
        assert g.nodes == set(names)
        # Each node should have at least some neighbors
        for name in names:
            assert len(g.neighbors(name)) > 0

    def test_small_world_tiny(self):
        # Less than 3 nodes falls back to complete
        names = ["a", "b"]
        g = SocialGraph.small_world(names, k=2, rng=random.Random(42))
        assert "b" in g.neighbors("a")
        assert "a" in g.neighbors("b")
