# Claude Code Automation Recommendations

Recommendations generated for happy-simulator based on codebase analysis.

## Codebase Profile

- **Type**: Python 3.13+ library (discrete-event simulation)
- **Tools**: pytest, mypy, ruff, mkdocs-material
- **Dependencies**: numpy, scipy, pandas, matplotlib
- **Size**: ~49 Python files, well-organized modular structure

---

## Hooks

### 1. Auto-format and lint on edit

**Why**: Ruff is configured with specific rules. Auto-formatting ensures consistent style.

**Where**: `.claude/settings.json`

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "ruff format $CLAUDE_FILE_PATHS && ruff check --fix $CLAUDE_FILE_PATHS"
          }
        ]
      }
    ]
  }
}
```

### 2. Type-check on edit

**Why**: Strict mypy is configured. Catching type errors immediately prevents regressions.

**Where**: `.claude/settings.json`

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "mypy $CLAUDE_FILE_PATHS --no-error-summary"
          }
        ]
      }
    ]
  }
}
```

---

## Skills

### 1. run-example

**Why**: Examples are substantial (e.g., `m_m_1_queue.py`, `metastable_state.py`). A skill to run and analyze simulation examples streamlines development.

**Create**: `.claude/skills/run-example/SKILL.md`

**Invocation**: User-only (`/run-example`)

```yaml
---
name: run-example
description: Run a simulation example and analyze the output
disable-model-invocation: true
---
# Run Example

Run one of the simulation examples from the `examples/` directory:
- dual_path_queue_latency.py
- increasing_queue_depth.py
- load_aware_routing.py
- m_m_1_queue.py
- metastable_state.py
- retrying_client.py

Execute with: `.venv/Scripts/python.exe examples/<name>.py`

Analyze the output and any generated plots. Report on simulation behavior.
```

### 2. gen-test

**Why**: Tests use specific patterns (ConstantArrivalTimeProvider, Instant.from_seconds()). A skill ensures consistency.

**Create**: `.claude/skills/gen-test/SKILL.md`

**Invocation**: User-only (`/gen-test`)

```yaml
---
name: gen-test
description: Generate a pytest test following project conventions
disable-model-invocation: true
---
# Generate Test

Create pytest tests following project patterns:
- Use `ConstantArrivalTimeProvider` for deterministic timing
- Use `Instant.from_seconds()` for time values
- Place unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Follow existing test naming: `test_<feature>.py`

Run tests with: `pytest -q` or `pytest tests/<file>.py -q`
```

---

## MCP Servers

### context7

**Why**: The project uses numpy, scipy, pandas, and matplotlib. Live documentation lookup helps with API details for these scientific computing libraries.

**Install**:

```bash
claude mcp add context7
```

---

## Subagents

### simulation-reviewer

**Why**: Discrete-event simulation has subtle correctness requirements (event ordering, time semantics, generator yields). A specialized reviewer catches issues.

**Create**: `.claude/agents/simulation-reviewer.md`

```markdown
# Simulation Reviewer

Review simulation code for:
- Correct Event invariants (target XOR callback)
- Proper use of Instant (nanoseconds internally, Instant.from_seconds() for creation)
- Generator semantics (yield delays correctly, return events properly)
- Source patterns (self-perpetuating, proper priming)
- Entity handle_event() implementations

Flag potential issues with:
- Time travel (events scheduled in the past)
- Infinite event loops
- Memory leaks from unprocessed generators
```

---

## Implementation Notes

To implement any of these recommendations:

1. **Hooks**: Merge the JSON into `.claude/settings.json` (or create `.claude/settings.local.json` for personal settings)
2. **Skills**: Create the directory structure and `SKILL.md` file as shown
3. **MCP Servers**: Run the install command shown
4. **Subagents**: Create the agent markdown file in `.claude/agents/`
