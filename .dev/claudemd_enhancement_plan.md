# CLAUDE.md Enhancement Plan

## Goal

Enhance CLAUDE.md to improve Claude's ability to understand and develop in this codebase, and establish conventions for using `.dev/` for design documentation.

---

## Part 1: Methodology for Project Understanding

When Claude first encounters this project, the following reading sequence provides optimal context-building:

### Reading Order for Claude

**Phase 1: Orientation (High-Level Understanding)**
1. `CLAUDE.md` - Primary entry point, project overview
2. `happysimulator/__init__.py` - Exported API surface, understand what's "public"

**Phase 2: Core Concepts (Foundational Understanding)**
3. `happysimulator/core/instant.py` - Time representation (fundamental)
4. `happysimulator/core/event.py` - Event structure and semantics
5. `happysimulator/core/entity.py` - Actor pattern and handle_event contract
6. `happysimulator/core/simulation.py` - Main simulation loop

**Phase 3: Patterns (How Things Connect)**
7. `happysimulator/load/source.py` - Event generation pattern
8. `happysimulator/components/queue.py` - Queue/Driver separation pattern
9. `happysimulator/components/queued_resource.py` - Resource pattern with capacity

**Phase 4: Examples (Practical Application)**
10. Read 1-2 examples (e.g., `examples/m_m_1_queue.py`) for full workflow understanding
11. Read corresponding integration tests for testing patterns

**Phase 5: Design Context (For New Features)**
12. `.dev/COMPONENTLIB.md` - Component design philosophy
13. `.dev/zipf-distribution-design.md` - Feature design template

### Mental Model Checklist

Claude should be able to answer these questions after reading:

- [ ] What is the fundamental unit of work? (Event)
- [ ] What are the two event handling styles? (target-based vs callback-based)
- [ ] How do generators enable multi-step processes?
- [ ] What is the Queue/QueueDriver separation pattern and why?
- [ ] How does time work? (Instant, nanoseconds internally, Epoch, Infinity)
- [ ] How do Sources generate load? (EventProvider + ArrivalTimeProvider)
- [ ] What are the component design principles? (composition, protocols, clock injection)

### Pattern Recognition Guide

| Pattern | Files to Reference | Key Characteristics |
|---------|-------------------|---------------------|
| Entity with capacity | `queue.py`, `queued_resource.py` | `has_capacity()` method, generator yields |
| Self-perpetuating events | `source.py`, `source_event.py` | Event schedules its successor |
| Clock injection | `entity.py`, `simulation.py` | `set_clock()` called during setup |
| Completion hooks | Look for `on_complete` callbacks | Loose coupling via callbacks |
| Policy pattern | `queue_policy.py` | Protocol-based, pluggable strategies |
| Distribution abstraction | `distributions/*.py` | `sample()` method, seedable |

---

## Part 2: Enhanced CLAUDE.md Structure

### Proposed New Structure

```
# CLAUDE.md

## Quick Reference                        [NEW]
## Project Overview                       [EXISTING - minor updates]
## Development Commands                   [EXISTING - keep as-is]
## Architecture                           [EXISTING - expand]
  ### Reading Order for New Contributors  [NEW]
  ### Core Abstractions                   [NEW]
  ### Component Design Principles         [NEW]
## Key Directories                        [EXISTING - expand]
## Patterns & Idioms                      [EXPANDED]
  ### Testing Patterns                    [NEW]
  ### Example Patterns                    [NEW]
## Code Style                             [EXISTING - minor additions]
## Troubleshooting                        [NEW]
## .dev Documentation Conventions         [NEW]
## Skills and Plugins                     [FIX - complete this section]
```

### Section Details

#### 1. Quick Reference (NEW)

Purpose: Enable Claude to get context in under 30 seconds for simple tasks.

```markdown
## Quick Reference

**What**: Discrete-event simulation library (like Matlab SimEvent)
**Core Loop**: EventHeap pop → Entity.handle_event() → schedule returned Events
**Key Invariant**: Events have EITHER `target` (Entity) OR `callback` (function)
**Time**: Use `Instant.from_seconds(n)`, not raw floats
**Generators**: Yield delays (float seconds); return events on completion
**Testing**: Use `ConstantArrivalTimeProvider` for deterministic timing
**Logging**: `$env:HS_LOGGING='DEBUG'` → `happysimulator.log`
```

#### 2. Reading Order for New Contributors (NEW)

```markdown
### Reading Order for New Contributors

**For understanding the simulation engine:**
1. `core/instant.py` → Time representation
2. `core/event.py` → Event structure
3. `core/entity.py` → Actor pattern
4. `core/simulation.py` → Main loop

**For adding new components:**
1. `components/queue.py` → Reference implementation
2. `components/queued_resource.py` → Simpler pattern
3. `.dev/COMPONENTLIB.md` → Design principles

**For understanding load generation:**
1. `load/source.py` → Event generator
2. `load/providers/*.py` → Arrival time strategies
3. `examples/m_m_1_queue.py` → Full example
```

#### 3. Core Abstractions (NEW)

```markdown
### Core Abstractions

#### Instant (`core/instant.py`)
- Internal representation: nanoseconds (int64)
- Creation: `Instant.from_seconds(1.5)`, `Instant.from_milliseconds(100)`
- Special values: `Instant.Epoch` (t=0), `Instant.Infinity` (termination)
- Arithmetic: `instant + seconds_float`, comparison operators work

#### Event (`core/event.py`)
- Immutable work unit scheduled on the EventHeap
- Required: `time` (Instant), `event_type` (str)
- Mutually exclusive: `target` (Entity) XOR `callback` (Callable)
- Optional: `context` (dict) for payload data

#### Entity (`core/entity.py`)
- Abstract base class for simulation actors
- Receives time via `set_clock()` during initialization
- Access current time via `self.now` property
- Override `handle_event(event)` to define behavior
- Override `has_capacity()` for resource constraints

#### Generator Semantics
- `yield delay_seconds` - pause this process
- `yield (delay, [side_effect_events])` - pause and emit side effects
- `return [events]` or `return event` - events on process completion
```

#### 4. Component Design Principles (NEW)

Extract key principles from COMPONENTLIB.md:

```markdown
### Component Design Principles

Components follow these established patterns:

1. **Composition over inheritance**: Build larger abstractions from smaller entities
2. **Protocol-based design**: Use `Simulatable` protocol for duck-typing
3. **Generator-friendly**: Express delays naturally with `yield`
4. **Clock injection**: Components receive time via `set_clock()`
5. **Completion hooks**: Enable loose coupling between components
6. **Transparent internals**: Hide implementation complexity from callers

Example structure:
```python
@dataclass
class MyComponent(Entity):
    name: str = "MyComponent"
    # Configuration parameters with sensible defaults

    # Private state with field(default=..., init=False)
    _internal_state: int = field(default=0, init=False)

    # Statistics tracking
    stats_processed: int = field(default=0, init=False)

    def has_capacity(self) -> bool:
        """Resource constraint check."""
        return self._internal_state < self.max_concurrent

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process with delays."""
        self._internal_state += 1
        yield self.processing_time
        self._internal_state -= 1
        self.stats_processed += 1
        return []
```

#### 5. Testing Patterns (NEW)

```markdown
## Testing Patterns

### Test Organization
- `tests/unit/` - Isolated component tests, no simulation loop
- `tests/integration/` - Full simulation tests with visualization output

### Deterministic Testing
Always use deterministic timing for reproducible tests:
```python
from happysimulator import ConstantArrivalTimeProvider, Instant

# NOT: PoissonArrivalTimeProvider (stochastic)
# USE: ConstantArrivalTimeProvider with fixed rate
arrival_provider = ConstantArrivalTimeProvider(
    ConstantRateProfile(rate=10.0),
    start_time=Instant.Epoch,
)
```

### Test Fixtures
- `test_output_dir` - Directory for test artifacts (plots, data)
- `timestamped_output_dir` - Keeps multiple runs for comparison

### Visualization Tests
```python
def test_with_visualization(self, test_output_dir: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # ... run simulation, collect data ...

    fig.savefig(test_output_dir / "my_plot.png", dpi=150)
    plt.close(fig)
```

### Statistical Verification
For probabilistic components (distributions, load balancing):
```python
# Sample many times, verify within tolerance
samples = [dist.sample() for _ in range(10000)]
observed_mean = sum(samples) / len(samples)
assert abs(observed_mean - expected_mean) < tolerance
```
```

#### 6. Example Patterns (NEW)

```markdown
## Example Patterns

Examples follow a consistent structure:

```python
"""One-line description.

Extended description explaining what this example demonstrates.

## Architecture

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Source  │─────►│  Queue  │─────►│ Server  │
└─────────┘      └─────────┘      └─────────┘
```

## Key Concepts
- Bullet points of what to learn
- Theory or formulas if applicable
"""

# =============================================================================
# Custom Entities for This Example
# =============================================================================

# ... entity definitions with docstrings ...

# =============================================================================
# Simulation Setup
# =============================================================================

def build_simulation() -> Simulation:
    """Construct the simulation model."""
    ...

# =============================================================================
# Visualization
# =============================================================================

def visualize_results(..., output_dir: Path):
    """Generate analysis plots."""
    ...

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "example_output" / Path(__file__).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = build_simulation()
    sim.run()
    visualize_results(..., output_dir)
```
```

#### 7. Troubleshooting (NEW)

```markdown
## Troubleshooting

### "Entity X is not attached to a simulation (Clock is None)"
**Cause**: Accessing `self.now` before the simulation calls `set_clock()`
**Fix**: Ensure entity is in the `entities` list passed to Simulation

### "Time travel detected" warnings
**Cause**: An event scheduled for a time before the current simulation time
**Fix**: Check that your entity's `handle_event` doesn't return events with times in the past
**Debug**: Set `$env:HS_LOGGING='DEBUG'` and check `happysimulator.log`

### Generator not completing / events missing
**Cause**: Generator yielded but never returned
**Fix**: Ensure your generator has a `return` statement (even if returning `[]`)

### Events processed past end_time
**Known Issue**: See TODO.md - loop condition checks time before popping
**Workaround**: Set `end_time` slightly beyond your actual end point

### Queue never drains
**Cause**: Usually a feedback loop issue - driver not receiving completion events
**Debug**: Add logging to `handle_event` to trace event flow
```

#### 8. .dev Documentation Conventions (NEW)

```markdown
## .dev Documentation Conventions

The `.dev/` directory contains design documents, implementation plans, and architectural decisions. These are internal documents not meant for end users.

### When to Create a .dev Document

Create a new document when:
- Designing a new major feature or component
- Making architectural decisions with trade-offs
- Planning multi-phase implementations
- Documenting design alternatives considered

### Document Template

```markdown
# [Feature Name] Design Document

## Overview
One-paragraph summary of what this document covers.

## Motivation
Why is this feature needed? What problem does it solve?
Include examples of current limitations.

## Requirements

### Functional Requirements
1. What the feature must do
2. Specific behaviors

### Non-Functional Requirements
1. Performance, usability, maintainability

## Design

### New Concepts
Classes, protocols, or abstractions introduced.

### Integration
How this integrates with existing code.

### File Organization
Where new files will live.

## Examples
Code showing how the feature is used.

## Testing Strategy
How to verify the feature works correctly.

## Alternatives Considered
Other approaches evaluated and why they were rejected.

## Implementation Plan
Ordered phases with milestones.

## References
Links to related work, papers, or prior art.
```

### Naming Conventions
- Feature designs: `feature-name-design.md` (kebab-case)
- Implementation plans: `FEATURE.md` (UPPER_SNAKE_CASE for major efforts)
- Decisions: `ADR-NNN-title.md` (Architecture Decision Records)

### Lifecycle
1. **Draft**: Initial design, open for feedback
2. **Approved**: Design accepted, ready for implementation
3. **Implemented**: Code complete, document serves as reference
4. **Superseded**: Replaced by newer design (link to successor)
```

#### 9. Skills and Plugins (FIX)

```markdown
## Claude Code Skills

The following skills are available when working with this codebase:

### Code Review
`/code-review` - Analyzes code changes for issues, patterns, and improvements

### Commit Commands
`/commit` - Creates well-formatted git commits following project conventions
`/commit-push-pr` - Commit, push, and open a pull request
`/clean_gone` - Clean up local branches deleted on remote

### Automation Recommender
`/claude-automation-recommender` - Analyze codebase and recommend Claude Code automations
```

---

## Part 3: Implementation Phases

### Phase 1: Structure and Quick Reference
1. Add "Quick Reference" section at the top of CLAUDE.md
2. Add "Reading Order for New Contributors" section
3. Reorganize existing sections for better flow

### Phase 2: Core Documentation
4. Expand "Architecture" with Core Abstractions subsection
5. Add Component Design Principles (extracted from COMPONENTLIB.md)
6. Expand "Key Directories" with component subdirectories detail

### Phase 3: Patterns and Testing
7. Expand "Common Patterns" with more examples
8. Add "Testing Patterns" section
9. Add "Example Patterns" section

### Phase 4: Troubleshooting and Meta
10. Add "Troubleshooting" section
11. Add ".dev Documentation Conventions" section
12. Fix "Skills and Plugins" section

### Phase 5: Validation
13. Review for consistency and completeness
14. Ensure no duplication with .dev documents
15. Verify all file paths are accurate

---

## Part 4: Files Reference

### File to Modify
- `CLAUDE.md` - Primary enhancement target

### Reference Files for Content
- `.dev/COMPONENTLIB.md` - Component design principles source
- `.dev/zipf-distribution-design.md` - Design document template reference
- `tests/conftest.py` - Test fixture patterns
- `examples/m_m_1_queue.py` - Example structure reference
- `happysimulator/core/entity.py` - Core abstraction reference
- `happysimulator/core/event.py` - Event system reference

---

## Verification

After implementation:
1. Read through enhanced CLAUDE.md for flow and completeness
2. Verify all referenced file paths exist
3. Confirm .dev conventions are clear and actionable
4. Test that a fresh Claude session can navigate the codebase using the new guidance
