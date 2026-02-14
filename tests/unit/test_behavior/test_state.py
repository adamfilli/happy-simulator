"""Unit tests for behavior state module."""

from happysimulator.components.behavior.state import AgentState, Memory


class TestMemory:
    def test_creation(self):
        m = Memory(time=1.0, event_type="Stimulus", source="env", valence=0.5)
        assert m.time == 1.0
        assert m.event_type == "Stimulus"
        assert m.valence == 0.5


class TestAgentState:
    def test_defaults(self):
        state = AgentState()
        assert state.satisfaction == 0.5
        assert state.energy == 1.0
        assert state.mood == 0.5
        assert state.beliefs == {}
        assert state.needs == {}
        assert state.knowledge == set()

    def test_add_memory(self):
        state = AgentState()
        for i in range(5):
            state.add_memory(Memory(time=float(i), event_type="E", valence=0.1 * i))
        assert len(state.recent_memories(10)) == 5

    def test_memory_capacity(self):
        state = AgentState()
        for i in range(150):
            state.add_memory(Memory(time=float(i), event_type="E"))
        # deque maxlen is 100
        assert len(state.recent_memories(200)) == 100

    def test_recent_memories_order(self):
        state = AgentState()
        state.add_memory(Memory(time=1.0, event_type="First"))
        state.add_memory(Memory(time=2.0, event_type="Second"))
        recent = state.recent_memories(2)
        assert recent[0].event_type == "Second"
        assert recent[1].event_type == "First"

    def test_average_recent_valence(self):
        state = AgentState()
        state.add_memory(Memory(time=1.0, event_type="E", valence=1.0))
        state.add_memory(Memory(time=2.0, event_type="E", valence=-1.0))
        assert state.average_recent_valence(2) == 0.0

    def test_average_recent_valence_empty(self):
        state = AgentState()
        assert state.average_recent_valence(5) == 0.0

    def test_decay_needs_drift_up(self):
        state = AgentState()
        state.needs["food"] = 0.0
        state.decay(10.0)  # 10 seconds
        assert state.needs["food"] == 0.1  # 0.01 * 10

    def test_decay_mood_toward_neutral(self):
        state = AgentState()
        state.mood = 0.9
        state.decay(5.0)
        assert state.mood < 0.9
        assert state.mood >= 0.5

    def test_decay_mood_from_low(self):
        state = AgentState()
        state.mood = 0.1
        state.decay(5.0)
        assert state.mood > 0.1
        assert state.mood <= 0.5

    def test_decay_energy_drift_down(self):
        state = AgentState()
        state.energy = 1.0
        state.decay(10.0)
        assert state.energy == 0.95  # 1.0 - 0.005 * 10

    def test_decay_zero_dt(self):
        state = AgentState()
        state.mood = 0.8
        state.decay(0.0)
        assert state.mood == 0.8

    def test_decay_negative_dt(self):
        state = AgentState()
        state.mood = 0.8
        state.decay(-1.0)
        assert state.mood == 0.8
