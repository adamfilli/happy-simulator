# User-Facing Claude Skills Design

> Shipped with the project so users of `happysimulator` get them automatically when using Claude Code.

## Motivation

The library has ~690 exported symbols, 350+ components, and 78 examples. New users face significant friction choosing the right abstractions, wiring them together, and interpreting results. These skills guide users through the full simulation lifecycle: **learn → build → debug → measure → understand**.

---

## Skills

### 1. `/scaffold` — Generate a new simulation from a description

**Why:** The biggest friction for new users is boilerplate. With ~690 exported symbols, knowing what to import and wire together is daunting. This skill asks "what do you want to simulate?" and generates a complete, runnable file.

**What it does:**
- Asks what system to simulate (if not specified)
- Generates entity classes, source, sink, simulation setup, and a `if __name__ == "__main__"` block
- Follows project conventions: `Instant.from_seconds()`, generator yields, `Event.once()` where appropriate
- Includes basic instrumentation (Sink, latency stats printout)
- Outputs a single `.py` file the user can run immediately

---

### 2. `/diagnose` — Troubleshoot a broken or misbehaving simulation

**Why:** Users hit the same gotchas repeatedly: missing entity registration (silent failure), queue growing forever, generators not progressing, wrong time types. The troubleshooting table in CLAUDE.md exists because these are real pain points.

**What it does:**
- Reads the user's simulation file(s)
- Checks for common mistakes:
  - Entities not registered in `Simulation(entities=[...])`
  - Missing `target` on Events
  - `has_capacity()` not overridden (queue never builds up)
  - Arrival rate > service rate (unbounded growth)
  - Raw floats instead of `Instant.from_seconds()`
  - Generator returning events instead of yielding them (or vice versa)
  - Source event double-counting
- Optionally runs the simulation and analyzes output
- Reports findings with fix suggestions

---

### 3. `/add-instrumentation` — Add observability to an existing simulation

**Why:** Users build the simulation first, then realize they need metrics. Adding probes, trackers, and charts requires knowing 3-4 different APIs (`Data`, `Probe`, `LatencyTracker`, `Chart`) and wiring them correctly. This is a natural "phase 2" action.

**What it does:**
- Reads the user's simulation code
- Identifies entities worth monitoring (QueuedResources, servers, sinks)
- Adds appropriate instrumentation:
  - `Probe` for queue depths and utilization
  - `LatencyTracker` / `Sink` for end-to-end latency
  - `ThroughputTracker` for throughput
  - `Data.bucket()` calls for time-series analysis
- Optionally adds `serve(sim, charts=[...])` for visual debugger
- Generates a matplotlib summary plot if visual deps aren't available

---

### 4. `/explain-example` — Walk through a library example with annotations

**Why:** There are 78 examples across 10 categories. They're excellent learning material but dense — a typical example is 200-400 lines with domain-specific patterns. Having Claude walk through one interactively is a much better learning experience than reading alone.

**What it does:**
- Lists example categories and files if none specified
- Reads the chosen example
- Explains it section by section: what entities do, how events flow, what the results demonstrate
- Highlights library patterns being used (generator yields, SimFuture, probes, etc.)
- Optionally runs the example and explains the output

---

### 5. `/component-guide` — Interactive component selection wizard

**Why:** The library has 350+ components across 15 sub-packages. Users know what they want to model (e.g., "a system with retries and timeouts") but not which components to use. This bridges the gap between the problem domain and the API.

**What it does:**
- Asks what the user is trying to model
- Recommends components from the library with rationale:
  - "For retry logic, use `CircuitBreaker` + `Fallback` from `resilience/`"
  - "For a manufacturing line, combine `ConveyorBelt` + `InspectionStation` + `BatchProcessor` from `industrial/`"
- Shows minimal wiring example for the recommended components
- Points to the most relevant example file(s) as reference

---

### 6. `/analyze` — Analyze simulation results using the built-in analysis API

**Why:** Users run simulations but struggle to interpret the output. The library has `analyze()`, `detect_phases()`, `Data.bucket()`, and `.to_prompt_context()` but users don't know these exist or how to chain them.

**What it does:**
- Takes a simulation file or summary data
- Runs/reads the simulation results
- Applies `detect_phases()` to find steady-state, warmup, and anomaly periods
- Buckets key metrics and identifies trends
- Uses `analyze()` + `.to_prompt_context()` for structured interpretation
- Produces a plain-English summary: "Queue stabilized at depth ~12 after warmup. P99 latency was 340ms. The system is running at 78% utilization — safe but close to the tipping point."

---

## Summary

| Skill | User Need | Lifecycle Phase |
|-------|-----------|-----------------|
| `/scaffold` | "I want to simulate X" | Starting a new simulation |
| `/diagnose` | "My simulation isn't working" | Debugging |
| `/add-instrumentation` | "I need metrics and charts" | After initial build |
| `/explain-example` | "How does this example work?" | Learning the library |
| `/component-guide` | "Which component should I use?" | Choosing components |
| `/analyze` | "What do these results mean?" | Interpreting output |

## Implementation Notes

- Each skill is a `SKILL.md` file in `.claude/skills/<skill-name>/`
- Skills should reference CLAUDE.md conventions (time types, generator yields, entity registration)
- Skills that run code should use the project's `.venv/Scripts/python.exe`
- `/scaffold` and `/add-instrumentation` produce code; others are primarily analytical
- All skills should follow the existing YAML frontmatter + Markdown instructions format
