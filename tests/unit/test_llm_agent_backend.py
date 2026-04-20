"""Tests for the LLM backend protocol and mock implementation."""

from happysimulator.components.llm_agent.backend import LLMBackend, MockLLMBackend


class TestMockLLMBackend:
    def test_returns_default_action(self):
        backend = MockLLMBackend(default_action="wait")
        result = backend.complete("What should I do?")
        assert result == "wait"

    def test_keyword_matching(self):
        backend = MockLLMBackend(
            responses={"hungry": "eat", "tired": "sleep"},
            default_action="wait",
        )
        assert backend.complete("I am hungry") == "eat"
        assert backend.complete("I am tired") == "sleep"
        assert backend.complete("I am fine") == "wait"

    def test_keyword_case_insensitive(self):
        backend = MockLLMBackend(responses={"FOOD": "eat"})
        assert backend.complete("I need food") == "eat"

    def test_call_count_tracking(self):
        backend = MockLLMBackend()
        assert backend.call_count == 0
        backend.complete("test")
        backend.complete("test")
        assert backend.call_count == 2

    def test_prompts_tracking(self):
        backend = MockLLMBackend()
        backend.complete("prompt one")
        backend.complete("prompt two")
        assert backend.prompts == ["prompt one", "prompt two"]

    def test_complete_structured_default(self):
        backend = MockLLMBackend(default_action="wait", default_reasoning="No reason.")
        result = backend.complete_structured("What?", schema={})
        assert result["action"] == "wait"
        assert result["reasoning"] == "No reason."

    def test_complete_structured_keyword_match_json(self):
        backend = MockLLMBackend(
            responses={"food": '{"action": "eat", "reasoning": "hungry"}'}
        )
        result = backend.complete_structured("need food", schema={})
        assert result["action"] == "eat"
        assert result["reasoning"] == "hungry"

    def test_complete_structured_keyword_match_non_json(self):
        backend = MockLLMBackend(responses={"food": "eat"})
        result = backend.complete_structured("need food", schema={})
        assert result["action"] == "eat"

    def test_model_id(self):
        backend = MockLLMBackend()
        assert backend.model_id == "mock-llm"


class TestLLMBackendProtocol:
    def test_mock_satisfies_protocol(self):
        backend = MockLLMBackend()
        assert isinstance(backend, LLMBackend)
