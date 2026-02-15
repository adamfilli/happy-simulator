"""Bank Run & Financial Contagion Simulation.

Models the classic Diamond-Dybvig bank run — a self-fulfilling prophecy where
individual rational behavior leads to collective catastrophe. A population of
200 depositors holds savings at a single bank. Confidence spreads through a
small-world social network. When confidence drops, depositors withdraw,
depleting reserves and eroding confidence further — a feedback loop that can
trigger cascading failure.

Three scenarios demonstrate different policy interventions:

  1. No intervention — pure bank run (self-fulfilling prophecy)
  2. Deposit insurance — eliminates the bad equilibrium
  3. Lender of last resort — arrests the run mid-cascade

Key library features demonstrated:

  - Behavioral agents with heterogeneous decision models
  - Social influence propagation (DeGroot model on small-world graph)
  - Custom entities (Bank, CentralBank) with feedback loops
  - Population segmentation (cautious / steady / loyal depositors)
  - Time-series data collection and multi-panel visualization

Usage:
    python examples/behavior/bank_run.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from happysimulator import Data, Event, Instant, Simulation, SimulationSummary
from happysimulator.components.behavior import (
    Agent,
    AgentState,
    DeGrootModel,
    DemographicSegment,
    Environment,
    NormalTraitDistribution,
    Population,
)
from happysimulator.components.behavior.decision import (
    BoundedRationalityModel,
    Choice,
    SocialInfluenceModel,
    UtilityModel,
)
from happysimulator.components.behavior.stimulus import (
    broadcast_stimulus,
    influence_propagation,
)
from happysimulator.core.entity import Entity
from happysimulator.instrumentation.probe import Probe

# =============================================================================
# Configuration & Result
# =============================================================================

SEGMENTS = ["cautious", "steady", "loyal"]


@dataclass
class BankRunConfig:
    """Parameters for a bank run scenario."""

    population_size: int = 200
    initial_deposit: float = 1000.0
    rumor_time: float = 10.0
    rumor_target_count: int = 5
    sim_duration: float = 90.0
    check_interval: float = 2.0
    deposit_insurance: bool = False
    lender_of_last_resort: bool = False
    liquidity_threshold: float = 0.5
    liquidity_injection_frac: float = 0.25
    seed: int = 42


@dataclass
class ScenarioResult:
    """Collected results from a single scenario run."""

    name: str
    summary: SimulationSummary
    health_data: Data
    confidence_data: dict[str, Data]
    withdrawal_data: dict[str, Data]
    final_health: float
    total_withdrawals: int
    withdrawals_by_segment: dict[str, int]
    survived: bool
    central_bank_injections: int = 0


# =============================================================================
# Bank Entity
# =============================================================================


class Bank(Entity):
    """Holds depositor reserves and processes withdrawals.

    Publishes health ratio to the Environment via StateChange events
    after each withdrawal, creating the feedback loop that drives contagion.
    """

    def __init__(self, name: str, total_deposits: float, env: Environment):
        super().__init__(name)
        self.total_deposits = total_deposits
        self.reserves = total_deposits
        self.env = env
        self.withdrawal_count = 0

    @property
    def health_ratio(self) -> float:
        if self.total_deposits <= 0:
            return 0.0
        return max(0.0, self.reserves / self.total_deposits)

    def handle_event(self, event):
        if event.event_type == "Withdraw":
            amount = event.context.get("metadata", {}).get("amount", 0)
            actual = min(amount, max(0.0, self.reserves))
            self.reserves -= actual
            self.withdrawal_count += 1
            return [
                Event(
                    time=self.now,
                    event_type="StateChange",
                    target=self.env,
                    context={
                        "metadata": {
                            "key": "bank_health",
                            "value": self.health_ratio,
                        }
                    },
                )
            ]

        if event.event_type == "Deposit":
            amount = event.context.get("metadata", {}).get("amount", 0)
            self.reserves += amount
            return [
                Event(
                    time=self.now,
                    event_type="StateChange",
                    target=self.env,
                    context={
                        "metadata": {
                            "key": "bank_health",
                            "value": self.health_ratio,
                        }
                    },
                )
            ]

        return None


# =============================================================================
# Central Bank Entity
# =============================================================================


class CentralBank(Entity):
    """Lender of last resort — injects emergency liquidity when reserves drop.

    Monitors bank health periodically and injects funds when the health
    ratio falls below the configured threshold.
    """

    def __init__(
        self,
        name: str,
        bank: Bank,
        threshold: float,
        injection_amount: float,
    ):
        super().__init__(name)
        self.bank = bank
        self.threshold = threshold
        self.injection_amount = injection_amount
        self.injections = 0

    def handle_event(self, event):
        if event.event_type == "MonitorBank" and self.bank.health_ratio < self.threshold:
            self.bank.reserves += self.injection_amount
            self.injections += 1
            return [
                Event(
                    time=self.now,
                    event_type="StateChange",
                    target=self.bank.env,
                    context={
                        "metadata": {
                            "key": "bank_health",
                            "value": self.bank.health_ratio,
                        }
                    },
                )
            ]
        return None


# =============================================================================
# Decision utility
# =============================================================================


def make_utility(insurance: bool = False):
    """Create the base utility function for withdraw/stay decisions.

    All three segments use this function, but wrapped in different decision
    model types (UtilityModel, SocialInfluenceModel, BoundedRationalityModel)
    which weight the output differently.

    Beliefs are stored in [-1, 1] range; mapped to [0, 1] confidence here.

    With deposit insurance, panic-driven withdrawals are essentially
    eliminated — depositors know their savings are guaranteed regardless
    of bank health, so there is no rational reason to run.
    """

    def utility(choice: Choice, ctx) -> float:
        bank_health = ctx.environment.get("bank_health", 1.0)
        neuroticism = ctx.traits.get("neuroticism")
        raw_belief = ctx.state.beliefs.get("bank_confidence", 0.6)
        confidence = max(0.0, min(1.0, (raw_belief + 1) / 2))

        if choice.action == "withdraw":
            if insurance:
                return 0.0  # deposits guaranteed — no rational panic
            health_fear = 1 - bank_health
            distrust = 1 - confidence
            panic = (health_fear * 0.4 + distrust * 0.6) * (0.5 + neuroticism)
            return min(1.0, panic)
        # stay
        trust = bank_health * 0.3 + confidence * 0.4
        calm = (1 - neuroticism) * 0.1
        score = trust + 0.05 + calm
        if insurance:
            score += 0.5  # insurance bonus
        return min(1.0, score)

    return utility


# =============================================================================
# Simulation builder
# =============================================================================


def _segment_ranges(pop_size: int) -> dict[str, range]:
    """Compute agent index ranges per segment (matches from_segments order)."""
    n_cautious = int(0.25 * pop_size)
    n_steady = int(0.55 * pop_size)
    return {
        "cautious": range(n_cautious),
        "steady": range(n_cautious, n_cautious + n_steady),
        "loyal": range(n_cautious + n_steady, pop_size),
    }


def run_scenario(name: str, config: BankRunConfig) -> ScenarioResult:
    """Build the simulation, schedule events, run, and collect results."""
    random.seed(config.seed)
    utility = make_utility(insurance=config.deposit_insurance)

    # --- Population segments ---
    segments = [
        DemographicSegment(
            name="cautious",
            fraction=0.25,
            trait_distribution=NormalTraitDistribution(
                means={
                    "openness": 0.4,
                    "conscientiousness": 0.5,
                    "extraversion": 0.4,
                    "agreeableness": 0.3,
                    "neuroticism": 0.8,
                },
                stds={"neuroticism": 0.08},
            ),
            decision_model_factory=lambda: UtilityModel(
                utility_fn=utility,
                temperature=0.0,
            ),
            initial_state_factory=lambda: AgentState(
                beliefs={"bank_confidence": 0.6},
            ),
            seed=config.seed,
        ),
        DemographicSegment(
            name="steady",
            fraction=0.55,
            trait_distribution=NormalTraitDistribution(
                means={
                    "openness": 0.5,
                    "conscientiousness": 0.5,
                    "extraversion": 0.5,
                    "agreeableness": 0.7,
                    "neuroticism": 0.5,
                },
            ),
            decision_model_factory=lambda: SocialInfluenceModel(
                individual_fn=utility,
                conformity_weight=0.6,
            ),
            initial_state_factory=lambda: AgentState(
                beliefs={"bank_confidence": 0.6},
            ),
            seed=config.seed + 1,
        ),
        DemographicSegment(
            name="loyal",
            fraction=0.20,
            trait_distribution=NormalTraitDistribution(
                means={
                    "openness": 0.5,
                    "conscientiousness": 0.8,
                    "extraversion": 0.4,
                    "agreeableness": 0.5,
                    "neuroticism": 0.2,
                },
                stds={"neuroticism": 0.08},
            ),
            decision_model_factory=lambda: BoundedRationalityModel(
                utility_fn=utility,
                aspiration=0.55,
            ),
            initial_state_factory=lambda: AgentState(
                beliefs={"bank_confidence": 0.6},
            ),
            seed=config.seed + 2,
        ),
    ]

    pop = Population.from_segments(
        total_size=config.population_size,
        segments=segments,
        graph_type="small_world",
        seed=config.seed,
    )

    # --- Segment bookkeeping ---
    seg_ranges = _segment_ranges(config.population_size)
    segment_map: dict[str, str] = {}
    for seg_name, idx_range in seg_ranges.items():
        for i in idx_range:
            segment_map[f"agent_{i}"] = seg_name

    # --- Bank & Environment ---
    total_deposits = config.population_size * config.initial_deposit

    env = Environment(
        name="economy",
        agents=pop.agents,
        social_graph=pop.social_graph,
        shared_state={"bank_health": 1.0},
        influence_model=DeGrootModel(self_weight=0.4),
        seed=config.seed,
    )

    bank = Bank("bank", total_deposits, env)

    # --- Optional Central Bank ---
    central_bank = None
    if config.lender_of_last_resort:
        injection = config.liquidity_injection_frac * total_deposits
        central_bank = CentralBank(
            "central_bank",
            bank,
            config.liquidity_threshold,
            injection,
        )

    # --- Tracking state ---
    withdrawn: set[str] = set()
    withdrawals_by_segment: dict[str, int] = dict.fromkeys(SEGMENTS, 0)
    confidence_data: dict[str, Data] = {s: Data() for s in SEGMENTS}
    withdrawal_timeline: dict[str, Data] = {s: Data() for s in SEGMENTS}
    insurance = config.deposit_insurance

    # --- Action handlers ---
    def withdraw_handler(ag: Agent, choice: Choice, event: Event):
        if ag.name in withdrawn:
            return None
        withdrawn.add(ag.name)
        seg = segment_map.get(ag.name, "unknown")
        withdrawals_by_segment[seg] = withdrawals_by_segment.get(seg, 0) + 1
        ag.state.beliefs["bank_confidence"] = -0.8
        return Event(
            time=ag.now,
            event_type="Withdraw",
            target=bank,
            context={"metadata": {"amount": config.initial_deposit}},
        )

    def stay_handler(ag: Agent, choice: Choice, event: Event):
        if ag.name in withdrawn:
            return
        health = env.shared_state.get("bank_health", 1.0)
        current = ag.state.beliefs.get("bank_confidence", 0.6)
        ag.state.beliefs["bank_confidence"] = 0.7 * current + 0.3 * (health * 2 - 1)
        if insurance:
            ag.state.beliefs["bank_confidence"] = max(
                ag.state.beliefs["bank_confidence"],
                -0.2,
            )
        return

    for agent in pop.agents:
        agent.on_action("withdraw", withdraw_handler)
        agent.on_action("stay", stay_handler)

    # --- Probe for bank health ---
    health_probe, health_data = Probe.on(bank, "health_ratio", interval=0.5)

    # --- Build simulation ---
    entities: list = [env, bank, *list(pop.agents)]
    if central_bank:
        entities.append(central_bank)

    sim = Simulation(
        end_time=Instant.from_seconds(config.sim_duration),
        entities=entities,
        probes=[health_probe],
    )

    # --- Schedule periodic CheckBank + influence rounds ---
    t = 1.0
    while t <= config.sim_duration:
        sim.schedule(influence_propagation(t, env, "bank_confidence"))

        if insurance:
            _schedule_confidence_clamp(sim, t + 0.005, pop.agents, withdrawn)

        sim.schedule(
            broadcast_stimulus(
                t + 0.01,
                env,
                "CheckBank",
                choices=["withdraw", "stay"],
            )
        )
        _schedule_confidence_sample(
            sim,
            t + 0.02,
            pop.agents,
            seg_ranges,
            confidence_data,
            withdrawal_timeline,
            withdrawals_by_segment,
        )
        t += config.check_interval

    # --- Rumor at t=rumor_time ---
    cautious_range = seg_ranges["cautious"]
    rng = random.Random(config.seed + 99)
    rumor_indices = rng.sample(
        list(cautious_range),
        min(config.rumor_target_count, len(cautious_range)),
    )
    rumor_target_names = {f"agent_{i}" for i in rumor_indices}

    def apply_rumor(event):
        for agent in pop.agents:
            if agent.name in rumor_target_names:
                agent.state.beliefs["bank_confidence"] = -0.4
        return []

    sim.schedule(
        Event.once(
            time=Instant.from_seconds(config.rumor_time - 0.05),
            event_type="Rumor",
            fn=apply_rumor,
        )
    )

    # --- Central bank monitoring ---
    if central_bank:
        monitor_t = config.rumor_time
        while monitor_t <= config.sim_duration:
            sim.schedule(
                Event(
                    time=Instant.from_seconds(monitor_t + 0.5),
                    event_type="MonitorBank",
                    target=central_bank,
                )
            )
            monitor_t += config.check_interval

    # --- Run ---
    summary = sim.run()

    return ScenarioResult(
        name=name,
        summary=summary,
        health_data=health_data,
        confidence_data=confidence_data,
        withdrawal_data=withdrawal_timeline,
        final_health=bank.health_ratio,
        total_withdrawals=len(withdrawn),
        withdrawals_by_segment=dict(withdrawals_by_segment),
        survived=bank.health_ratio > 0.05,
        central_bank_injections=(central_bank.injections if central_bank else 0),
    )


def _schedule_confidence_clamp(
    sim: Simulation,
    time: float,
    agents: list[Agent],
    withdrawn: set[str],
) -> None:
    """After influence propagation, clamp confidence for insured depositors."""

    def clamp(event):
        for agent in agents:
            if agent.name not in withdrawn:
                agent.state.beliefs["bank_confidence"] = max(
                    agent.state.beliefs.get("bank_confidence", 0.6),
                    -0.2,
                )
        return []

    sim.schedule(
        Event.once(
            time=Instant.from_seconds(time),
            event_type="ClampConfidence",
            fn=clamp,
        )
    )


def _schedule_confidence_sample(
    sim: Simulation,
    time: float,
    agents: list[Agent],
    seg_ranges: dict[str, range],
    confidence_data: dict[str, Data],
    withdrawal_timeline: dict[str, Data],
    withdrawals_by_segment: dict[str, int],
) -> None:
    """Schedule a callback that samples average confidence per segment."""

    def sample(event):
        for seg_name, idx_range in seg_ranges.items():
            seg_agents = [agents[i] for i in idx_range]
            if seg_agents:
                avg_conf = sum(
                    a.state.beliefs.get("bank_confidence", 0.6) for a in seg_agents
                ) / len(seg_agents)
                confidence_data[seg_name].add_stat(avg_conf, event.time)
            withdrawal_timeline[seg_name].add_stat(
                withdrawals_by_segment.get(seg_name, 0),
                event.time,
            )
        return []

    sim.schedule(
        Event.once(
            time=Instant.from_seconds(time),
            event_type="SampleConfidence",
            fn=sample,
        )
    )


# =============================================================================
# Output
# =============================================================================


def print_results(results: list[ScenarioResult]) -> None:
    """Print a comparison table of all scenario results."""
    pop_size = 200
    print("=" * 78)
    print("  BANK RUN & FINANCIAL CONTAGION SIMULATION")
    print("  Diamond-Dybvig model with heterogeneous depositors")
    print("=" * 78)

    for r in results:
        status = "SURVIVED" if r.survived else "FAILED"
        print(f"\n  Scenario: {r.name}  [{status}]")
        print(f"  {'-' * 72}")
        print(f"    Final bank health:     {r.final_health:6.1%}")
        print(f"    Total withdrawals:     {r.total_withdrawals:>4d} / {pop_size}")
        for seg in SEGMENTS:
            count = r.withdrawals_by_segment.get(seg, 0)
            print(f"      {seg:>10s}:          {count:>4d}")
        if r.central_bank_injections > 0:
            print(f"    CB interventions:      {r.central_bank_injections}")
        print(f"    Events processed:      {r.summary.total_events_processed:,d}")
        print(f"    Wall-clock time:       {r.summary.wall_clock_seconds:.2f}s")

    print(f"\n{'=' * 78}")
    print(f"  {'SCENARIO COMPARISON':^74s}")
    print(f"{'=' * 78}")
    print(
        f"  {'Scenario':<28s} {'Survived':>10s} {'Health':>8s} {'Withdrawals':>12s} {'CB Inj.':>8s}"
    )
    print(f"  {'-' * 28} {'-' * 10} {'-' * 8} {'-' * 12} {'-' * 8}")
    for r in results:
        status = "Yes" if r.survived else "No"
        print(
            f"  {r.name:<28s} {status:>10s} {r.final_health:>7.1%} "
            f"{r.total_withdrawals:>12d} {r.central_bank_injections:>8d}"
        )


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(results: list[ScenarioResult], output_dir: Path) -> None:
    """Generate a 4-panel chart comparing all scenarios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    seg_colors = {"cautious": "#ef4444", "steady": "#f59e0b", "loyal": "#22c55e"}
    scenario_colors = ["#ef4444", "#3b82f6", "#22c55e"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Bank Health over time (all scenarios) ---
    ax = axes[0, 0]
    for i, r in enumerate(results):
        times = r.health_data.times()
        values = r.health_data.raw_values()
        if times:
            ax.plot(times, values, linewidth=2, label=r.name, color=scenario_colors[i])
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="CB threshold")
    ax.axvline(x=10.0, color="gray", linestyle="--", alpha=0.4)
    ax.annotate("Rumor", xy=(10, 0.95), fontsize=8, color="gray", ha="center")
    ax.set_ylabel("Health Ratio")
    ax.set_title("Bank Health (Reserves / Total Deposits)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Average Confidence by Segment (no-intervention scenario) ---
    ax = axes[0, 1]
    r0 = results[0]
    for seg in SEGMENTS:
        data = r0.confidence_data.get(seg)
        if data and data.times():
            conf_mapped = [(v + 1) / 2 for v in data.raw_values()]
            ax.plot(data.times(), conf_mapped, linewidth=1.5, label=seg, color=seg_colors[seg])
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=10.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Confidence (0-1)")
    ax.set_title(f"Agent Confidence — {r0.name}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Cumulative Withdrawals by Segment (no-intervention) ---
    ax = axes[1, 0]
    for seg in SEGMENTS:
        data = r0.withdrawal_data.get(seg)
        if data and data.times():
            ax.fill_between(data.times(), data.raw_values(), alpha=0.3, color=seg_colors[seg])
            ax.plot(
                data.times(), data.raw_values(), linewidth=1.5, label=seg, color=seg_colors[seg]
            )
    ax.axvline(x=10.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative Withdrawals")
    ax.set_title(f"Withdrawal Cascade — {r0.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Scenario Comparison Bar Chart ---
    ax = axes[1, 1]
    names = [r.name.replace(". ", ".\n") for r in results]
    healths = [r.final_health for r in results]
    bars = ax.bar(names, healths, color=scenario_colors[: len(results)], alpha=0.85)
    for bar, h in zip(bars, healths, strict=False):
        label = f"{h:.0%}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Final Health Ratio")
    ax.set_title("Scenario Comparison")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Bank Run & Financial Contagion — Diamond-Dybvig Model",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "bank_run.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    scenarios = [
        ("No intervention", BankRunConfig(seed=42)),
        ("Deposit insurance", BankRunConfig(seed=42, deposit_insurance=True)),
        ("Lender of last resort", BankRunConfig(seed=42, lender_of_last_resort=True)),
    ]

    results: list[ScenarioResult] = []
    for scenario_name, config in scenarios:
        print(f"\n  Running: {scenario_name} ...")
        result = run_scenario(scenario_name, config)
        results.append(result)

    print_results(results)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("\n  [matplotlib not installed, skipping charts]")
        return

    output_dir = Path("output/bank_run")
    print(f"\n  Generating visualization -> {output_dir.absolute()}")
    visualize_results(results, output_dir)
    print(f"\n  Done. Charts saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
