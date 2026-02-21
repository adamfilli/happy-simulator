"""Tests for sanity checks."""

from happysimulator.components.llm_agent.sanity import (
    NoDoubleEating,
    NoSleepWhileSleeping,
    run_checks,
)
from happysimulator.components.llm_agent.state import HumanState


class TestNoDoubleEating:
    def test_allows_first_meal(self):
        check = NoDoubleEating()
        state = HumanState()
        result = check.check(state, "eat", {"current_time": 100.0})
        assert result is None

    def test_detects_double_eating(self):
        check = NoDoubleEating()
        state = HumanState()
        result = check.check(
            state, "eat", {"current_time": 200.0, "last_eat_time": 100.0}
        )
        assert result is not None
        assert "1800" in result

    def test_allows_after_enough_time(self):
        check = NoDoubleEating()
        state = HumanState()
        result = check.check(
            state, "eat", {"current_time": 3000.0, "last_eat_time": 100.0}
        )
        assert result is None

    def test_ignores_non_eat_actions(self):
        check = NoDoubleEating()
        state = HumanState()
        result = check.check(
            state, "sleep", {"current_time": 200.0, "last_eat_time": 100.0}
        )
        assert result is None


class TestNoSleepWhileSleeping:
    def test_allows_sleep_when_awake(self):
        check = NoSleepWhileSleeping()
        state = HumanState()
        result = check.check(state, "sleep", {"is_sleeping": False})
        assert result is None

    def test_detects_double_sleep(self):
        check = NoSleepWhileSleeping()
        state = HumanState()
        result = check.check(state, "sleep", {"is_sleeping": True})
        assert result is not None
        assert "already sleeping" in result.lower()

    def test_ignores_non_sleep_actions(self):
        check = NoSleepWhileSleeping()
        state = HumanState()
        result = check.check(state, "eat", {"is_sleeping": True})
        assert result is None


class TestRunChecks:
    def test_empty_checks_pass(self):
        state = HumanState()
        failures = run_checks([], state, "eat", {})
        assert failures == []

    def test_aggregates_failures(self):
        checks = [NoDoubleEating(), NoSleepWhileSleeping()]
        state = HumanState()
        # Eating too soon AND sleeping while sleeping (hypothetical)
        context = {
            "current_time": 200.0,
            "last_eat_time": 100.0,
            "is_sleeping": True,
        }
        # Only NoDoubleEating triggers for "eat"
        failures = run_checks(checks, state, "eat", context)
        assert len(failures) == 1

    def test_all_pass(self):
        checks = [NoDoubleEating(), NoSleepWhileSleeping()]
        state = HumanState()
        failures = run_checks(checks, state, "wait", {})
        assert failures == []
