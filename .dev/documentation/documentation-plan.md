# Documentation Plan for happy-simulator

## Context

The project has 200+ public API symbols, 78 examples, and excellent docstring coverage (95%+), but the existing docs are 3 minimal pages (index, getting-started, api.md) that are **outdated** — they reference a removed `callback` parameter. MkDocs Material is already configured with GitHub Pages CI/CD, but no auto-generation from docstrings is set up. The goal is to create first-class documentation with progressive user guides and comprehensive API reference.

## Approach

- Stay with **MkDocs Material** (already deployed, CI/CD working)
- Add **mkdocstrings[python]** for auto-generated API reference from existing docstrings
- Adapt guide content from **CLAUDE.md** (23KB of structured examples) — not writing from scratch
- ~63 doc files total: 3 top-level + 16 guides + 33 API reference + 11 examples

---

## Phase 1: Infrastructure Setup

### 1.1 Update `pyproject.toml`
Add `mkdocstrings[python]>=0.25` to dev dependencies.

### 1.2 Rewrite `mkdocs.yml`
Full config with mkdocstrings, full nav structure, Material theme features.

### 1.3 Update `.github/workflows/docs.yml`
Add `pip install -e .` so mkdocstrings can import modules. Use `mkdocs build --strict`.

### 1.4 Create directory skeleton

---

## Phase 2: Core Guides (highest impact)

| File | Content |
|------|---------|
| `docs/index.md` | Hero, feature cards, minimal example |
| `docs/installation.md` | PyPI, source, extras |
| `guides/getting-started.md` | First simulation walkthrough |
| `guides/core-concepts.md` | Instant/Duration, Event, Entity, Simulation |
| `guides/generators-and-futures.md` | yield forms, SimFuture, any_of/all_of |
| `guides/load-generation.md` | Source factories, profiles |

---

## Phase 3: API Reference (auto-generated)

Each page has a brief intro + `:::` mkdocstrings directives.

### Core (9 pages)
simulation, event, entity, temporal, sim-future, clock, logical-clocks, protocols, control

### Components (27 pages)
One page per sub-package.

### Other packages (9 pages)
load, distributions, instrumentation, analysis, sketching, faults, visual, ai, logging

---

## Phase 4: Remaining Guides (12 pages)

queuing-and-resources, observability, simulation-control, visual-debugger, networking, clocks, distributed-systems, behavioral-modeling, industrial-simulation, fault-injection, logging, testing-patterns

---

## Phase 5: Examples Gallery + Design

11 example category pages + 1 design philosophy page.

---

## Verification

1. `pip install -e ".[dev]"` — install package + docs deps
2. `mkdocs serve` — local preview
3. `mkdocs build --strict` — verify no broken refs
4. CI: push to branch, verify GitHub Actions workflow passes
