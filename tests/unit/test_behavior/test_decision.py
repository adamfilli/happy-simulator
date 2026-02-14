"""Unit tests for behavior decision module."""

import random

from happysimulator.components.behavior.traits import PersonalityTraits
from happysimulator.components.behavior.state import AgentState
from happysimulator.components.behavior.decision import (
    Choice,
    DecisionContext,
    UtilityModel,
    Rule,
    RuleBasedModel,
    BoundedRationalityModel,
    SocialInfluenceModel,
    CompositeModel,
)


def _make_context(choices, traits=None, social_context=None):
    return DecisionContext(
        traits=traits or PersonalityTraits.big_five(),
        state=AgentState(),
        choices=choices,
        social_context=social_context or {},
    )


class TestUtilityModel:
    def test_deterministic_argmax(self):
        def utility(choice, ctx):
            return {"buy": 0.9, "wait": 0.1}.get(choice.action, 0)

        model = UtilityModel(utility_fn=utility)
        choices = [Choice(action="buy"), Choice(action="wait")]
        ctx = _make_context(choices)

        result = model.decide(ctx, random.Random(42))
        assert result is not None
        assert result.action == "buy"

    def test_softmax_stochastic(self):
        def utility(choice, ctx):
            return {"a": 0.5, "b": 0.5}.get(choice.action, 0)

        model = UtilityModel(utility_fn=utility, temperature=1.0)
        choices = [Choice(action="a"), Choice(action="b")]
        ctx = _make_context(choices)

        # Both should be selected over many trials
        actions = set()
        for seed in range(100):
            r = model.decide(ctx, random.Random(seed))
            if r:
                actions.add(r.action)
        assert "a" in actions and "b" in actions

    def test_empty_choices(self):
        model = UtilityModel(utility_fn=lambda c, ctx: 1.0)
        ctx = _make_context([])
        assert model.decide(ctx, random.Random(42)) is None


class TestRuleBasedModel:
    def test_first_matching_rule(self):
        rules = [
            Rule(condition=lambda ctx: ctx.state.mood > 0.7, action="celebrate", priority=1),
            Rule(condition=lambda ctx: True, action="wait", priority=0),
        ]
        model = RuleBasedModel(rules)
        state = AgentState()
        state.mood = 0.8
        choices = [Choice(action="celebrate"), Choice(action="wait")]
        ctx = DecisionContext(
            traits=PersonalityTraits.big_five(),
            state=state,
            choices=choices,
        )
        result = model.decide(ctx, random.Random(42))
        assert result is not None
        assert result.action == "celebrate"

    def test_default_action(self):
        rules = [
            Rule(condition=lambda ctx: False, action="never"),
        ]
        model = RuleBasedModel(rules, default_action="fallback")
        choices = [Choice(action="never"), Choice(action="fallback")]
        ctx = _make_context(choices)
        result = model.decide(ctx, random.Random(42))
        assert result is not None
        assert result.action == "fallback"

    def test_no_matching_rule_no_default(self):
        rules = [Rule(condition=lambda ctx: False, action="never")]
        model = RuleBasedModel(rules)
        choices = [Choice(action="never")]
        ctx = _make_context(choices)
        assert model.decide(ctx, random.Random(42)) is None


class TestBoundedRationalityModel:
    def test_satisficing(self):
        def utility(choice, ctx):
            return {"good": 0.7, "great": 0.9, "bad": 0.1}.get(choice.action, 0)

        model = BoundedRationalityModel(utility_fn=utility, aspiration=0.6)
        choices = [Choice(action="bad"), Choice(action="good"), Choice(action="great")]
        ctx = _make_context(choices)

        # Should pick either good or great (both exceed 0.6)
        result = model.decide(ctx, random.Random(42))
        assert result is not None
        assert result.action in ("good", "great")

    def test_nothing_above_aspiration_picks_best(self):
        def utility(choice, ctx):
            return {"a": 0.3, "b": 0.4}.get(choice.action, 0)

        model = BoundedRationalityModel(utility_fn=utility, aspiration=0.9)
        choices = [Choice(action="a"), Choice(action="b")]
        ctx = _make_context(choices)

        result = model.decide(ctx, random.Random(42))
        assert result is not None
        assert result.action == "b"


class TestSocialInfluenceModel:
    def test_with_peer_actions(self):
        def utility(choice, ctx):
            return 0.5

        model = SocialInfluenceModel(individual_fn=utility, conformity_weight=0.8)
        choices = [Choice(action="buy"), Choice(action="wait")]

        # All peers chose buy
        ctx = _make_context(
            choices,
            traits=PersonalityTraits.big_five(agreeableness=1.0),
            social_context={"peer_actions": {"buy": 100, "wait": 0}},
        )

        buy_count = 0
        for seed in range(100):
            r = model.decide(ctx, random.Random(seed))
            if r and r.action == "buy":
                buy_count += 1
        # Should strongly favor buy
        assert buy_count > 70


class TestCompositeModel:
    def test_weighted_voting(self):
        model_a = UtilityModel(
            utility_fn=lambda c, ctx: 1.0 if c.action == "a" else 0.0
        )
        model_b = UtilityModel(
            utility_fn=lambda c, ctx: 1.0 if c.action == "b" else 0.0
        )
        composite = CompositeModel([(model_a, 2.0), (model_b, 1.0)])
        choices = [Choice(action="a"), Choice(action="b")]
        ctx = _make_context(choices)
        result = composite.decide(ctx, random.Random(42))
        assert result is not None
        assert result.action == "a"  # Higher weight

    def test_empty_choices(self):
        composite = CompositeModel([])
        ctx = _make_context([])
        assert composite.decide(ctx, random.Random(42)) is None
