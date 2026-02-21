"""Tests for the state-to-prompt translation."""

from happysimulator.components.llm_agent.memory import EpisodicMemory, MemoryEntry
from happysimulator.components.llm_agent.prompt import (
    PromptBuilder,
    SalienceThreshold,
    _intensity_level,
)
from happysimulator.components.llm_agent.state import HumanState


class TestIntensityLevel:
    def test_low_intensity(self):
        assert _intensity_level(0.45, 0.4) == "low"

    def test_medium_intensity(self):
        assert _intensity_level(0.65, 0.4) == "medium"

    def test_high_intensity(self):
        assert _intensity_level(0.95, 0.4) == "high"


class TestPromptBuilder:
    def test_omits_sub_threshold_state(self):
        builder = PromptBuilder("a calm person")
        state = HumanState()
        # All defaults are sub-threshold
        state.physiology.hunger = 0.1
        state.emotion.anger = 0.1

        desc = builder.describe_state(state)
        assert "hungry" not in desc.lower()
        assert "anger" not in desc.lower()

    def test_includes_salient_state(self):
        builder = PromptBuilder("a stressed person")
        state = HumanState()
        state.physiology.hunger = 0.8  # Above 0.4 threshold

        desc = builder.describe_state(state)
        assert "hungry" in desc.lower()

    def test_intensity_appropriate_language(self):
        builder = PromptBuilder("a person")
        state = HumanState()

        # Low hunger (just above threshold)
        state.physiology.hunger = 0.45
        desc_low = builder.describe_state(state)
        assert "getting hungry" in desc_low.lower()

        # High hunger
        state.physiology.hunger = 0.95
        desc_high = builder.describe_state(state)
        assert "starving" in desc_high.lower()

    def test_full_prompt_includes_persona(self):
        builder = PromptBuilder("Dr. Smith, a cautious doctor")
        state = HumanState()
        memory = EpisodicMemory()
        memory.add(MemoryEntry(time=1.0, summary="Patient arrived"))

        prompt = builder.build_decision_prompt(
            state, "A patient needs help", memory, ["treat", "wait", "refer"]
        )

        assert "Dr. Smith" in prompt
        assert "Patient arrived" in prompt
        assert "treat" in prompt
        assert "wait" in prompt

    def test_prompt_includes_social_context(self):
        builder = PromptBuilder("an employee")
        state = HumanState()
        memory = EpisodicMemory()

        prompt = builder.build_decision_prompt(
            state,
            "Boss asked for the report",
            memory,
            ["comply", "delay"],
            social_context={"Boss": {"trust": 0.8, "liking": 0.3}},
        )

        assert "Boss" in prompt
        assert "trust" in prompt

    def test_prompt_with_custom_thresholds(self):
        thresholds = [
            SalienceThreshold(
                "hunger",
                0.9,  # Very high threshold
                {"low": "Slightly peckish.", "medium": "Hungry.", "high": "Starving."},
            ),
        ]
        builder = PromptBuilder("a person", thresholds=thresholds)
        state = HumanState()
        state.physiology.hunger = 0.5

        desc = builder.describe_state(state)
        assert desc == ""  # Below custom threshold

    def test_prompt_includes_available_actions(self):
        builder = PromptBuilder("a person")
        state = HumanState()
        memory = EpisodicMemory()

        prompt = builder.build_decision_prompt(
            state, "Something happened", memory, ["run", "hide", "fight"]
        )

        assert "run" in prompt
        assert "hide" in prompt
        assert "fight" in prompt
        assert "Choose one action" in prompt
