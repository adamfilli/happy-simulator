"""Unit tests for the CausalGraph and build_causal_graph."""

import pytest

from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.recorder import InMemoryTraceRecorder
from happysimulator.analysis.causal_graph import (
    CausalGraph,
    CausalNode,
    build_causal_graph,
)


# ---------------------------------------------------------------------------
# CausalNode
# ---------------------------------------------------------------------------

class TestCausalNode:
    def test_to_dict(self):
        node = CausalNode(
            event_id="abc",
            event_type="Request",
            time=Instant.from_seconds(1.5),
            parent_id="parent-1",
        )
        d = node.to_dict()
        assert d["event_id"] == "abc"
        assert d["event_type"] == "Request"
        assert d["time_s"] == pytest.approx(1.5)
        assert d["parent_id"] == "parent-1"

    def test_to_dict_root(self):
        node = CausalNode("r", "Init", Instant.Epoch, None)
        assert node.to_dict()["parent_id"] is None

    def test_frozen(self):
        node = CausalNode("a", "X", Instant.Epoch, None)
        with pytest.raises(AttributeError):
            node.event_id = "b"


# ---------------------------------------------------------------------------
# CausalGraph — empty / single
# ---------------------------------------------------------------------------

class TestCausalGraphBasic:
    def test_empty_graph(self):
        g = CausalGraph({})
        assert len(g) == 0
        assert g.roots() == []
        assert g.leaves() == []
        assert g.critical_path() == []
        assert "x" not in g

    def test_single_root(self):
        n = CausalNode("a", "Start", Instant.Epoch, None)
        g = CausalGraph({"a": n})

        assert len(g) == 1
        assert "a" in g
        assert g.roots() == [n]
        assert g.leaves() == [n]
        assert g.parent("a") is None
        assert g.children("a") == []
        assert g.ancestors("a") == []
        assert g.descendants("a") == []
        assert g.depth("a") == 0
        assert g.critical_path() == [n]


# ---------------------------------------------------------------------------
# CausalGraph — linear chain A → B → C
# ---------------------------------------------------------------------------

class TestCausalGraphLinearChain:
    @pytest.fixture
    def chain_graph(self):
        a = CausalNode("a", "Start", Instant.from_seconds(0), None)
        b = CausalNode("b", "Mid", Instant.from_seconds(1), "a")
        c = CausalNode("c", "End", Instant.from_seconds(2), "b")
        return CausalGraph({"a": a, "b": b, "c": c})

    def test_roots(self, chain_graph):
        roots = chain_graph.roots()
        assert len(roots) == 1
        assert roots[0].event_id == "a"

    def test_leaves(self, chain_graph):
        leaves = chain_graph.leaves()
        assert len(leaves) == 1
        assert leaves[0].event_id == "c"

    def test_parent(self, chain_graph):
        assert chain_graph.parent("c").event_id == "b"
        assert chain_graph.parent("b").event_id == "a"
        assert chain_graph.parent("a") is None

    def test_children(self, chain_graph):
        assert [n.event_id for n in chain_graph.children("a")] == ["b"]
        assert [n.event_id for n in chain_graph.children("b")] == ["c"]
        assert chain_graph.children("c") == []

    def test_ancestors(self, chain_graph):
        anc = chain_graph.ancestors("c")
        assert [n.event_id for n in anc] == ["b", "a"]

    def test_descendants(self, chain_graph):
        desc = chain_graph.descendants("a")
        assert [n.event_id for n in desc] == ["b", "c"]

    def test_depth(self, chain_graph):
        assert chain_graph.depth("a") == 0
        assert chain_graph.depth("b") == 1
        assert chain_graph.depth("c") == 2

    def test_critical_path(self, chain_graph):
        path = chain_graph.critical_path()
        assert [n.event_id for n in path] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# CausalGraph — tree: A → {B, C}
# ---------------------------------------------------------------------------

class TestCausalGraphTree:
    @pytest.fixture
    def tree_graph(self):
        a = CausalNode("a", "Root", Instant.from_seconds(0), None)
        b = CausalNode("b", "Left", Instant.from_seconds(1), "a")
        c = CausalNode("c", "Right", Instant.from_seconds(1), "a")
        return CausalGraph({"a": a, "b": b, "c": c})

    def test_roots_and_leaves(self, tree_graph):
        assert len(tree_graph.roots()) == 1
        assert len(tree_graph.leaves()) == 2

    def test_children_of_root(self, tree_graph):
        children = tree_graph.children("a")
        assert {n.event_id for n in children} == {"b", "c"}

    def test_depth(self, tree_graph):
        assert tree_graph.depth("a") == 0
        assert tree_graph.depth("b") == 1
        assert tree_graph.depth("c") == 1


# ---------------------------------------------------------------------------
# Multiple roots
# ---------------------------------------------------------------------------

class TestCausalGraphMultipleRoots:
    def test_independent_families(self):
        r1 = CausalNode("r1", "Root1", Instant.from_seconds(0), None)
        c1 = CausalNode("c1", "Child1", Instant.from_seconds(1), "r1")
        r2 = CausalNode("r2", "Root2", Instant.from_seconds(0.5), None)
        c2 = CausalNode("c2", "Child2", Instant.from_seconds(1.5), "r2")

        g = CausalGraph({"r1": r1, "c1": c1, "r2": r2, "c2": c2})

        roots = g.roots()
        assert len(roots) == 2
        # Sorted by time
        assert roots[0].event_id == "r1"
        assert roots[1].event_id == "r2"

        assert g.descendants("r1") == [g.nodes["c1"]]
        assert g.descendants("r2") == [g.nodes["c2"]]


# ---------------------------------------------------------------------------
# Critical path selection
# ---------------------------------------------------------------------------

class TestCriticalPath:
    def test_picks_longest_branch(self):
        """A → B → C → D (depth 3) vs A → E (depth 1)."""
        a = CausalNode("a", "A", Instant.from_seconds(0), None)
        b = CausalNode("b", "B", Instant.from_seconds(1), "a")
        c = CausalNode("c", "C", Instant.from_seconds(2), "b")
        d = CausalNode("d", "D", Instant.from_seconds(3), "c")
        e = CausalNode("e", "E", Instant.from_seconds(1), "a")

        g = CausalGraph({"a": a, "b": b, "c": c, "d": d, "e": e})
        path = g.critical_path()
        assert [n.event_id for n in path] == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# filter() with parent re-linking
# ---------------------------------------------------------------------------

class TestCausalGraphFilter:
    def test_filter_keeps_matching(self):
        a = CausalNode("a", "Request", Instant.from_seconds(0), None)
        b = CausalNode("b", "Internal", Instant.from_seconds(1), "a")
        c = CausalNode("c", "Request", Instant.from_seconds(2), "b")

        g = CausalGraph({"a": a, "b": b, "c": c})

        # Keep only Request events
        filtered = g.filter(lambda n: n.event_type == "Request")
        assert len(filtered) == 2
        assert "a" in filtered
        assert "c" in filtered
        assert "b" not in filtered

    def test_filter_relinks_parents(self):
        """A(Request) → B(Internal) → C(Request): C's parent re-links to A."""
        a = CausalNode("a", "Request", Instant.from_seconds(0), None)
        b = CausalNode("b", "Internal", Instant.from_seconds(1), "a")
        c = CausalNode("c", "Request", Instant.from_seconds(2), "b")

        g = CausalGraph({"a": a, "b": b, "c": c})
        filtered = g.filter(lambda n: n.event_type == "Request")

        # C should now point to A (skipping B)
        assert filtered.parent("c").event_id == "a"
        assert filtered.children("a")[0].event_id == "c"

    def test_filter_root_stays_root(self):
        """When the parent chain is fully excluded, node becomes root."""
        a = CausalNode("a", "Internal", Instant.from_seconds(0), None)
        b = CausalNode("b", "Request", Instant.from_seconds(1), "a")

        g = CausalGraph({"a": a, "b": b})
        filtered = g.filter(lambda n: n.event_type == "Request")

        assert len(filtered) == 1
        assert filtered.parent("b") is None
        assert filtered.roots()[0].event_id == "b"


# ---------------------------------------------------------------------------
# build_causal_graph() from recorder spans
# ---------------------------------------------------------------------------

class TestBuildCausalGraph:
    def _make_schedule_span(self, event_id, event_type, time_s, parent_id=None):
        """Helper to create a simulation.schedule span dict."""
        return {
            "time": Instant.from_seconds(time_s),
            "kind": "simulation.schedule",
            "event_id": event_id,
            "event_type": event_type,
            "data": {
                "scheduled_time": Instant.from_seconds(time_s),
                "parent_id": parent_id,
            },
        }

    def test_basic_build(self):
        recorder = InMemoryTraceRecorder()
        recorder.spans = [
            self._make_schedule_span("a", "Request", 0.0),
            self._make_schedule_span("b", "Response", 1.0, parent_id="a"),
        ]

        graph = build_causal_graph(recorder)
        assert len(graph) == 2
        assert graph.parent("b").event_id == "a"

    def test_deduplication(self):
        """ProcessContinuation reuses event_id; only first occurrence kept."""
        recorder = InMemoryTraceRecorder()
        recorder.spans = [
            self._make_schedule_span("a", "Request", 0.0),
            self._make_schedule_span("a", "Request", 0.1),  # duplicate
        ]

        graph = build_causal_graph(recorder)
        assert len(graph) == 1

    def test_exclude_event_types(self):
        recorder = InMemoryTraceRecorder()
        recorder.spans = [
            self._make_schedule_span("a", "Request", 0.0),
            self._make_schedule_span("b", "Probe", 0.5),
            self._make_schedule_span("c", "Response", 1.0, parent_id="a"),
        ]

        graph = build_causal_graph(recorder, exclude_event_types={"Probe"})
        assert len(graph) == 2
        assert "b" not in graph

    def test_ignores_non_schedule_spans(self):
        recorder = InMemoryTraceRecorder()
        recorder.spans = [
            {
                "time": Instant.Epoch,
                "kind": "simulation.init",
                "data": {"num_sources": 1},
            },
            self._make_schedule_span("a", "Request", 0.0),
        ]

        graph = build_causal_graph(recorder)
        assert len(graph) == 1

    def test_missing_event_id_skipped(self):
        recorder = InMemoryTraceRecorder()
        recorder.spans = [
            {
                "time": Instant.Epoch,
                "kind": "simulation.schedule",
                "event_type": "Orphan",
            },
        ]

        graph = build_causal_graph(recorder)
        assert len(graph) == 0


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

class TestCausalGraphToDict:
    def test_to_dict_stats(self):
        a = CausalNode("a", "Root", Instant.from_seconds(0), None)
        b = CausalNode("b", "Child", Instant.from_seconds(1), "a")

        g = CausalGraph({"a": a, "b": b})
        d = g.to_dict()

        assert d["stats"]["total_nodes"] == 2
        assert d["stats"]["roots"] == 1
        assert d["stats"]["leaves"] == 1
        assert d["stats"]["max_depth"] == 1
        assert d["stats"]["critical_path_length"] == 2
        assert len(d["nodes"]) == 2
