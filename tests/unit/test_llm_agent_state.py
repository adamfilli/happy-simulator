"""Tests for the LLM agent state system."""

from happysimulator.components.llm_agent.state import (
    CognitiveState,
    EmotionalState,
    HumanState,
    PhysiologicalState,
    SocialRelation,
)


class TestPhysiologicalState:
    def test_hunger_increases_over_time(self):
        state = PhysiologicalState(hunger=0.0)
        state.tick(3600.0, 12.0)  # 1 hour at noon
        assert state.hunger > 0.0
        assert abs(state.hunger - 0.12) < 0.01

    def test_fatigue_increases_over_time(self):
        state = PhysiologicalState(fatigue=0.0)
        state.tick(3600.0, 12.0)
        assert state.fatigue > 0.0
        assert abs(state.fatigue - 0.06) < 0.01

    def test_pain_decays_exponentially(self):
        state = PhysiologicalState(pain=0.8)
        state.tick(3600.0, 12.0)  # 1 hour
        assert state.pain < 0.8
        # Half-life ~1 hour, so roughly 0.4
        assert 0.3 < state.pain < 0.5

    def test_hunger_clamped_at_one(self):
        state = PhysiologicalState(hunger=0.95)
        state.tick(3600.0, 12.0)
        assert state.hunger <= 1.0

    def test_pain_near_zero_rounds_to_zero(self):
        state = PhysiologicalState(pain=0.0005)
        state.tick(3600.0, 12.0)
        assert state.pain == 0.0

    def test_comfort_changes_with_circadian(self):
        state = PhysiologicalState(comfort=0.5)
        state.tick(3600.0, 10.0)  # Peak alertness hour
        # Comfort should drift toward circadian target
        assert state.comfort != 0.5


class TestEmotionalState:
    def test_positive_event_increases_mood_and_joy(self):
        state = EmotionalState(mood=0.5, joy=0.0)
        state.apply_event_impact(valence=0.5, arousal_delta=0.1)
        assert state.mood > 0.5
        assert state.joy > 0.0
        assert state.arousal > 0.3

    def test_negative_event_increases_anger_and_anxiety(self):
        state = EmotionalState(mood=0.5, anger=0.0, anxiety=0.0)
        state.apply_event_impact(valence=-0.5, arousal_delta=0.1)
        assert state.mood < 0.5
        assert state.anger > 0.0
        assert state.anxiety > 0.0

    def test_decay_reduces_emotions(self):
        state = EmotionalState(arousal=0.8, anger=0.5, anxiety=0.5, joy=0.5)
        state.decay(3600.0)  # 1 hour
        assert state.arousal < 0.8
        assert state.anger < 0.5
        assert state.anxiety < 0.5
        assert state.joy < 0.5

    def test_mood_drifts_toward_baseline(self):
        state = EmotionalState(mood=0.9)
        state.decay(3600.0)
        assert state.mood < 0.9

    def test_values_clamped(self):
        state = EmotionalState(mood=0.0)
        state.apply_event_impact(valence=-1.0, arousal_delta=0.0)
        assert state.mood >= 0.0


class TestCognitiveState:
    def test_effective_capacity_fresh(self):
        state = CognitiveState(attention=1.0, decision_fatigue=0.0)
        assert state.effective_capacity == 1.0

    def test_effective_capacity_depleted(self):
        state = CognitiveState(attention=0.5, decision_fatigue=0.5)
        assert state.effective_capacity == 0.25

    def test_record_decision_increases_fatigue(self):
        state = CognitiveState(decision_fatigue=0.0)
        state.record_decision()
        assert state.decision_fatigue > 0.0
        assert state.attention < 1.0

    def test_rest_restores_capacity(self):
        state = CognitiveState(attention=0.5, decision_fatigue=0.5)
        state.rest(3600.0)
        assert state.attention > 0.5
        assert state.decision_fatigue < 0.5

    def test_effective_capacity_clamped(self):
        state = CognitiveState(attention=0.0, decision_fatigue=1.0)
        assert state.effective_capacity == 0.0


class TestSocialRelation:
    def test_defaults(self):
        rel = SocialRelation()
        assert rel.trust == 0.5
        assert rel.familiarity == 0.0
        assert rel.liking == 0.0
        assert rel.debt == 0.0


class TestHumanState:
    def test_tick_delegates_to_substates(self):
        state = HumanState()
        state.physiology.hunger = 0.0
        state.emotion.arousal = 0.5
        state.cognition.attention = 0.8

        state.tick(3600.0, 12.0)

        assert state.physiology.hunger > 0.0
        assert state.emotion.arousal < 0.5
        assert state.cognition.attention > 0.8  # rest restores

    def test_get_relation_creates_new(self):
        state = HumanState()
        rel = state.get_relation("Bob")
        assert isinstance(rel, SocialRelation)
        assert "Bob" in state.social

    def test_get_relation_returns_existing(self):
        state = HumanState()
        state.social["Alice"] = SocialRelation(trust=0.9)
        rel = state.get_relation("Alice")
        assert rel.trust == 0.9

    def test_snapshot_keys(self):
        state = HumanState()
        snap = state.snapshot()
        expected_keys = {
            "hunger", "fatigue", "pain", "comfort",
            "mood", "arousal", "anger", "anxiety", "joy",
            "attention", "decision_fatigue", "effective_capacity",
        }
        assert set(snap.keys()) == expected_keys
