"""Tests for LLM agent package exports."""


class TestLLMAgentPackageExports:
    def test_import_from_llm_agent_package(self):
        from happysimulator.components.llm_agent import (
            CognitiveState,
            DecisionTrace,
            EmotionalState,
            EpisodicMemory,
            HeuristicLoop,
            HumanAgent,
            HumanAgentStats,
            HumanState,
            LLMBackend,
            MemoryEntry,
            MockLLMBackend,
            NoDoubleEating,
            NoSleepWhileSleeping,
            PhysiologicalState,
            PromptBuilder,
            ReactiveLoop,
            ReasoningLoop,
            SalienceThreshold,
            SanityCheck,
            SocialRelation,
            run_checks,
            select_loop,
        )
        # Smoke test: all imported successfully
        assert HumanAgent is not None

    def test_import_from_components(self):
        from happysimulator.components import (
            HumanAgent,
            HumanAgentStats,
            HumanState,
            MockLLMBackend,
        )
        assert HumanAgent is not None

    def test_import_from_top_level(self):
        from happysimulator import (
            HumanAgent,
            HumanAgentStats,
            HumanState,
            LLMBackend,
            MockLLMBackend,
        )
        assert HumanAgent is not None
        assert isinstance(MockLLMBackend(), LLMBackend)
