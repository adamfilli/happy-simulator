"""Human state system for LLM-powered agents.

Pure-code state that evolves with simulation time. No LLM calls.
Each dimension is a float in [0, 1] (or [-1, 1] for liking) with
explicit semantics. State ticks are driven by simulation dt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PhysiologicalState:
    """Physiological needs that change passively with time.

    Attributes:
        hunger: 0=full, 1=starving. Ramps ~0.12/hour.
        fatigue: 0=rested, 1=exhausted. Ramps ~0.06/hour awake.
        circadian_phase: Personal phase offset in radians.
        pain: 0=none, 1=severe. Exponential decay.
        comfort: 0=miserable, 1=comfortable.
    """

    hunger: float = 0.0
    fatigue: float = 0.0
    circadian_phase: float = 0.0
    pain: float = 0.0
    comfort: float = 0.8

    def tick(self, dt_seconds: float, time_of_day_hours: float) -> None:
        """Advance physiological state by dt seconds."""
        dt_hours = dt_seconds / 3600.0

        # Hunger ramps linearly ~0.12/hour
        self.hunger = min(1.0, self.hunger + 0.12 * dt_hours)

        # Fatigue ramps ~0.06/hour while awake
        self.fatigue = min(1.0, self.fatigue + 0.06 * dt_hours)

        # Pain decays exponentially (half-life ~1 hour)
        if self.pain > 0.001:
            decay = math.exp(-0.693 * dt_hours)
            self.pain = max(0.0, self.pain * decay)
        else:
            self.pain = 0.0

        # Circadian alertness modulates comfort slightly
        # Peak alertness ~10am (hour 10), trough ~3am (hour 3)
        phase = (time_of_day_hours + self.circadian_phase) * (2 * math.pi / 24.0)
        circadian_factor = 0.5 + 0.5 * math.cos(phase - 10.0 * (2 * math.pi / 24.0))
        # Comfort drifts toward circadian baseline
        target_comfort = 0.5 + 0.3 * circadian_factor
        drift_rate = 0.1 * dt_hours
        self.comfort = self.comfort + drift_rate * (target_comfort - self.comfort)
        self.comfort = max(0.0, min(1.0, self.comfort))


@dataclass
class EmotionalState:
    """Emotional dimensions with event-driven changes and decay.

    Attributes:
        mood: 0=negative, 1=positive. Baseline 0.5.
        arousal: 0=calm, 1=extremely activated.
        anger: 0=none, 1=extreme.
        anxiety: 0=none, 1=extreme.
        joy: 0=none, 1=extreme.
    """

    mood: float = 0.5
    arousal: float = 0.3
    anger: float = 0.0
    anxiety: float = 0.0
    joy: float = 0.0

    def apply_event_impact(self, valence: float, arousal_delta: float) -> None:
        """Shift emotions in response to an event.

        Args:
            valence: Positive shifts mood/joy up, negative shifts anger up.
            arousal_delta: Direct arousal change.
        """
        self.mood = max(0.0, min(1.0, self.mood + valence * 0.15))
        self.arousal = max(0.0, min(1.0, self.arousal + arousal_delta))
        if valence > 0:
            self.joy = min(1.0, self.joy + valence * 0.2)
        else:
            self.anger = min(1.0, self.anger + abs(valence) * 0.15)
            self.anxiety = min(1.0, self.anxiety + abs(valence) * 0.1)

    def decay(self, dt_seconds: float) -> None:
        """Decay emotions toward baseline over dt seconds."""
        dt_hours = dt_seconds / 3600.0

        # Mood drifts toward 0.5 (slow)
        self.mood += 0.1 * dt_hours * (0.5 - self.mood)

        # Arousal decays fast (half-life ~30 min)
        self.arousal *= math.exp(-1.386 * dt_hours)

        # Anger decays medium (half-life ~1 hour)
        self.anger *= math.exp(-0.693 * dt_hours)

        # Anxiety decays slow (half-life ~2 hours)
        self.anxiety *= math.exp(-0.347 * dt_hours)

        # Joy decays medium (half-life ~1 hour)
        self.joy *= math.exp(-0.693 * dt_hours)

        # Clamp near-zero values
        if self.arousal < 0.001:
            self.arousal = 0.0
        if self.anger < 0.001:
            self.anger = 0.0
        if self.anxiety < 0.001:
            self.anxiety = 0.0
        if self.joy < 0.001:
            self.joy = 0.0


@dataclass
class CognitiveState:
    """Cognitive capacity and load.

    Attributes:
        attention: 0=depleted, 1=full.
        decision_fatigue: 0=fresh, 1=depleted.
        working_memory_load: 0=free, 1=overloaded.
    """

    attention: float = 1.0
    decision_fatigue: float = 0.0
    working_memory_load: float = 0.0

    def record_decision(self) -> None:
        """Record that a decision was made, increasing fatigue."""
        self.decision_fatigue = min(1.0, self.decision_fatigue + 0.05)
        self.attention = max(0.0, self.attention - 0.02)

    def rest(self, dt_seconds: float) -> None:
        """Restore cognitive resources over time."""
        dt_hours = dt_seconds / 3600.0
        self.attention = min(1.0, self.attention + 0.2 * dt_hours)
        self.decision_fatigue = max(0.0, self.decision_fatigue - 0.1 * dt_hours)

    @property
    def effective_capacity(self) -> float:
        """Effective cognitive capacity considering fatigue and attention.

        Returns:
            Float in [0, 1] representing available cognitive capacity.
        """
        base = self.attention * (1.0 - self.decision_fatigue)
        return max(0.0, min(1.0, base))


@dataclass
class SocialRelation:
    """Per-relationship state, asymmetric.

    Attributes:
        trust: 0=no trust, 1=full trust.
        familiarity: 0=stranger, 1=very familiar.
        liking: -1=dislike, 1=like.
        debt: Obligation balance (positive = they owe me).
    """

    trust: float = 0.5
    familiarity: float = 0.0
    liking: float = 0.0
    debt: float = 0.0


@dataclass
class HumanState:
    """Composite state combining all dimensions.

    Attributes:
        physiology: Physiological needs.
        emotion: Emotional state.
        cognition: Cognitive capacity.
        social: Per-agent relationship state.
        beliefs: Explicit beliefs as key-value pairs.
    """

    physiology: PhysiologicalState = field(default_factory=PhysiologicalState)
    emotion: EmotionalState = field(default_factory=EmotionalState)
    cognition: CognitiveState = field(default_factory=CognitiveState)
    social: dict[str, SocialRelation] = field(default_factory=dict)
    beliefs: dict[str, str] = field(default_factory=dict)

    def tick(self, dt_seconds: float, time_of_day_hours: float) -> None:
        """Tick all sub-states forward by dt seconds."""
        self.physiology.tick(dt_seconds, time_of_day_hours)
        self.emotion.decay(dt_seconds)
        self.cognition.rest(dt_seconds)

    def get_relation(self, name: str) -> SocialRelation:
        """Get or create a social relation for the given agent name."""
        if name not in self.social:
            self.social[name] = SocialRelation()
        return self.social[name]

    def snapshot(self) -> dict[str, object]:
        """Return a serializable snapshot of salient state dimensions."""
        return {
            "hunger": self.physiology.hunger,
            "fatigue": self.physiology.fatigue,
            "pain": self.physiology.pain,
            "comfort": self.physiology.comfort,
            "mood": self.emotion.mood,
            "arousal": self.emotion.arousal,
            "anger": self.emotion.anger,
            "anxiety": self.emotion.anxiety,
            "joy": self.emotion.joy,
            "attention": self.cognition.attention,
            "decision_fatigue": self.cognition.decision_fatigue,
            "effective_capacity": self.cognition.effective_capacity,
        }
