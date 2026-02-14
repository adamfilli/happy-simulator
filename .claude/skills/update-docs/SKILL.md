---
name: update-docs
description: Review changes on the current branch and update documentation to reflect new or changed functionality
---

# Update Documentation

Review changes on the current branch (vs `main`) and update the MkDocs documentation site to reflect new or changed functionality.

## Documentation Architecture

The docs use **MkDocs Material** with **mkdocstrings** for auto-generated API reference. Config is in `mkdocs.yml`.

### What auto-updates (no action needed)

API reference pages in `docs/reference/` use `:::` mkdocstrings directives that render docs directly from source docstrings. When you add/change docstrings in source code, the API reference updates automatically on next build. **No manual edits needed for API reference pages unless a new module is added.**

### What requires manual updates

| Doc Area | Location | When to Update |
|----------|----------|----------------|
| **Guides** | `docs/guides/*.md` | New features, changed APIs, new patterns |
| **API reference pages** | `docs/reference/**/*.md` | New modules/packages added |
| **Examples gallery** | `docs/examples/*.md` | New example files added |
| **mkdocs.yml nav** | `mkdocs.yml` | New pages added |
| **Landing page** | `docs/index.md` | Major new capabilities |
| **Installation** | `docs/installation.md` | New dependencies or extras |

## Instructions

### 1. Identify what changed

```bash
# Files changed on this branch vs main
git diff --name-only main...HEAD

# Detailed diff for source code changes
git diff main...HEAD -- happysimulator/

# New example files
git diff --name-only --diff-filter=A main...HEAD -- examples/

# Changes to public API
git diff main...HEAD -- happysimulator/__init__.py
```

### 2. Determine which docs need updates

Check each category:

**New module/package added?**
→ Create a new `docs/reference/<area>/<name>.md` page with a `:::` directive
→ Add it to the `nav:` section in `mkdocs.yml`

**New component or feature?**
→ Update the relevant guide in `docs/guides/` (e.g., new queue policy → update `queuing-and-resources.md`)
→ If it's a major new capability, consider whether it needs its own guide page

**Changed API (renamed, new params, removed)?**
→ Update any guide that references the old API
→ Docstring changes auto-propagate to reference pages

**New example file added?**
→ Add a row to the relevant `docs/examples/<category>.md` gallery page
→ Update the count in `docs/examples/index.md`

**New dependencies or extras?**
→ Update `docs/installation.md`

### 3. Apply updates

For each doc file that needs changes:

1. **Read the existing file** to understand its structure
2. **Make minimal, targeted edits** that match the existing style
3. **Keep code examples correct** — no references to removed APIs (e.g., the old `callback=` parameter)
4. **Maintain cross-links** between guides and reference pages

### 4. Verify

```bash
python -m mkdocs build
```

Check for errors. Warnings about docstring formatting in source files are acceptable; warnings about missing pages or broken nav entries are not.

## Guide File Reference

| Guide | Covers |
|-------|--------|
| `guides/getting-started.md` | First simulation, Source→Server→Sink, M/M/1 |
| `guides/core-concepts.md` | Instant, Duration, Event, Entity, Simulation, clock injection |
| `guides/generators-and-futures.md` | yield forms, SimFuture, any_of, all_of |
| `guides/load-generation.md` | Source factories, profiles, custom providers |
| `guides/queuing-and-resources.md` | Queue, QueuedResource, Resource, rate limiters, Inductor |
| `guides/observability.md` | Data, Probe, collectors, SimulationSummary, analysis |
| `guides/simulation-control.md` | pause/resume, stepping, breakpoints, hooks |
| `guides/visual-debugger.md` | serve(), Chart, transforms, UI features |
| `guides/networking.md` | Network, links, conditions, partitions |
| `guides/clocks.md` | NodeClock, FixedSkew, LinearDrift, logical clocks |
| `guides/distributed-systems.md` | Raft, Paxos, CRDTs, replication, locks |
| `guides/behavioral-modeling.md` | Agent, Population, Environment, decisions, influence |
| `guides/industrial-simulation.md` | Industrial component catalog, composition |
| `guides/fault-injection.md` | FaultSchedule, breakdowns, partitions |
| `guides/logging.md` | Console/file/JSON logging, env vars |
| `guides/testing-patterns.md` | Deterministic testing, fixtures, seeds |

## API Reference Page Template

When adding a new reference page for a new module:

```markdown
# Module Name

Brief one-sentence description of what this module provides.

::: happysimulator.path.to.module
```

## Example Gallery Entry Template

When adding a new example to a gallery page:

```markdown
| [example_name.py](https://github.com/adamfilli/happy-simulator/blob/main/examples/category/example_name.py) | One-line description |
```

## Output

Provide a summary of what was updated:

```
## Changes Analyzed
- <list of relevant changes on this branch>

## Documentation Updates
- <what was added/changed in docs and why>

## No Updates Needed
- <areas checked that don't need changes>
```

If no documentation updates are needed, explain why (e.g., "internal refactoring with no API changes").
