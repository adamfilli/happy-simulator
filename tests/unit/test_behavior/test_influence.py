"""Unit tests for behavior influence module."""

import random

from happysimulator.components.behavior.influence import (
    DeGrootModel,
    BoundedConfidenceModel,
    VoterModel,
    InfluenceModel,
)


class TestDeGrootModel:
    def test_convergence_toward_neighbors(self):
        model = DeGrootModel(self_weight=0.5)
        current = 0.0
        opinions = [1.0, 1.0]
        weights = [1.0, 1.0]
        result = model.compute_influence(current, opinions, weights, random.Random(42))
        # self_weight * 0.0 + 0.5 * avg(1.0, 1.0) = 0.5
        assert abs(result - 0.5) < 1e-9

    def test_empty_influencers(self):
        model = DeGrootModel()
        result = model.compute_influence(0.7, [], [], random.Random(42))
        assert result == 0.7

    def test_self_weight_one(self):
        model = DeGrootModel(self_weight=1.0)
        result = model.compute_influence(0.3, [0.9], [1.0], random.Random(42))
        assert abs(result - 0.3) < 1e-9

    def test_protocol_conformance(self):
        model = DeGrootModel()
        assert isinstance(model, InfluenceModel)


class TestBoundedConfidenceModel:
    def test_filters_distant_opinions(self):
        model = BoundedConfidenceModel(epsilon=0.2, self_weight=0.0)
        current = 0.5
        opinions = [0.6, 0.9]  # 0.9 is too far (|0.9-0.5| > 0.2)
        weights = [1.0, 1.0]
        result = model.compute_influence(current, opinions, weights, random.Random(42))
        # Only 0.6 is within epsilon
        assert abs(result - 0.6) < 1e-9

    def test_all_too_far(self):
        model = BoundedConfidenceModel(epsilon=0.01, self_weight=0.5)
        result = model.compute_influence(0.5, [0.0, 1.0], [1.0, 1.0], random.Random(42))
        assert result == 0.5

    def test_all_close(self):
        model = BoundedConfidenceModel(epsilon=1.0, self_weight=0.5)
        result = model.compute_influence(0.5, [0.4, 0.6], [1.0, 1.0], random.Random(42))
        expected = 0.5 * 0.5 + 0.5 * 0.5  # self_weight * 0.5 + 0.5 * avg(0.4, 0.6)
        assert abs(result - expected) < 1e-9


class TestVoterModel:
    def test_adopts_one_neighbor(self):
        model = VoterModel()
        opinions = [0.1, 0.9]
        weights = [1.0, 1.0]
        results = set()
        for seed in range(100):
            r = model.compute_influence(0.5, opinions, weights, random.Random(seed))
            results.add(round(r, 1))
        # Should adopt either 0.1 or 0.9
        assert 0.1 in results
        assert 0.9 in results

    def test_empty_influencers(self):
        model = VoterModel()
        result = model.compute_influence(0.5, [], [], random.Random(42))
        assert result == 0.5

    def test_single_neighbor(self):
        model = VoterModel()
        result = model.compute_influence(0.5, [0.8], [1.0], random.Random(42))
        assert result == 0.8
