"""Unit tests for behavior traits module."""

import random

from happysimulator.components.behavior.traits import (
    NormalTraitDistribution,
    PersonalityTraits,
    TraitSet,
    UniformTraitDistribution,
)


class TestPersonalityTraits:
    def test_big_five_defaults(self):
        traits = PersonalityTraits.big_five()
        assert traits.get("openness") == 0.5
        assert traits.get("neuroticism") == 0.5
        assert len(traits.names()) == 5

    def test_big_five_custom_values(self):
        traits = PersonalityTraits.big_five(openness=0.9, agreeableness=0.1)
        assert traits.get("openness") == 0.9
        assert traits.get("agreeableness") == 0.1
        assert traits.get("extraversion") == 0.5

    def test_get_missing_dimension_returns_default(self):
        traits = PersonalityTraits(dimensions={"a": 0.3})
        assert traits.get("a") == 0.3
        assert traits.get("missing") == 0.5

    def test_frozen(self):
        traits = PersonalityTraits.big_five()
        assert isinstance(traits, PersonalityTraits)
        # frozen dataclass
        try:
            traits.dimensions = {}  # type: ignore
            raise AssertionError("Should raise")
        except AttributeError:
            pass

    def test_clamping(self):
        traits = PersonalityTraits.big_five(openness=1.5, neuroticism=-0.5)
        assert traits.get("openness") == 1.0
        assert traits.get("neuroticism") == 0.0

    def test_protocol_conformance(self):
        traits = PersonalityTraits.big_five()
        assert isinstance(traits, TraitSet)


class TestNormalTraitDistribution:
    def test_deterministic_with_seed(self):
        dist = NormalTraitDistribution(
            means={"a": 0.5, "b": 0.7},
            stds={"a": 0.1, "b": 0.1},
        )
        rng = random.Random(42)
        t1 = dist.sample(rng)

        rng2 = random.Random(42)
        t2 = dist.sample(rng2)

        assert t1.get("a") == t2.get("a")
        assert t1.get("b") == t2.get("b")

    def test_values_clamped(self):
        dist = NormalTraitDistribution(means={"x": 0.0}, stds={"x": 10.0})
        rng = random.Random(1)
        for _ in range(50):
            t = dist.sample(rng)
            assert 0.0 <= t.get("x") <= 1.0


class TestUniformTraitDistribution:
    def test_dimensions(self):
        dist = UniformTraitDistribution(["a", "b", "c"])
        rng = random.Random(42)
        traits = dist.sample(rng)
        assert set(traits.names()) == {"a", "b", "c"}
        for name in traits.names():
            assert 0.0 <= traits.get(name) <= 1.0

    def test_deterministic(self):
        dist = UniformTraitDistribution(["x"])
        t1 = dist.sample(random.Random(99))
        t2 = dist.sample(random.Random(99))
        assert t1.get("x") == t2.get("x")
