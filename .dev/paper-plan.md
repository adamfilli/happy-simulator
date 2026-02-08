# Plan: Academic Paper for happy-simulator

## Context

happy-simulator is a discrete-event simulation library (Python 3.13+, v0.1.4, Apache 2.0) designed for software engineers to model distributed systems and observe emergent behavior before deployment. The project has 118 Python files, 186 public API items, 50+ pre-built components, 36+ integration tests, and 16 runnable examples. The goal is to turn this into a publishable academic paper -- the author's first.

The library fills an unserved niche: DES tools designed for software engineers (not operations researchers), with batteries-included distributed-systems primitives (caches, circuit breakers, rate limiters, queue policies, streaming algorithms).

---

## Publication Strategy

**Recommended: Two-stage approach**

### Stage 1: JOSS (Journal of Open Source Software) -- Start here
- **Format**: ~1,000 words + references (short software description)
- **Review**: Transparent GitHub-based review; constructive, focused on software quality
- **Timeline**: 2-3 weeks writing, 4-8 weeks review = ~3 months to publication
- **Why first**: Fastest path to a citable DOI. The project already meets JOSS requirements (installable, tested, documented, licensed, examples). Low barrier for a first-time author.
- **Deliverable**: `paper.md` + `paper.bib` added to the repository

### Stage 2: SoftwareX (Elsevier) -- Expand later
- **Format**: 3,000-6,000 words with case studies and evaluation
- **Timeline**: 4-6 weeks writing, 3-6 months review
- **Why second**: Allows deeper technical presentation of architecture and case studies. Can reference the JOSS DOI. No conflict -- JOSS is a software description, SoftwareX is a research paper.

**Note**: Winter Simulation Conference (WSC) is aspirational but competitive and requires framing for simulation researchers specifically. Can be a third target after the first two.

---

## What I'll Write (Deliverable)

I will create a **full JOSS paper draft** (`paper.md` + `paper.bib`) plus a **detailed SoftwareX paper outline** that can be expanded into a full draft later.

### Files to create:
1. **`paper/joss/paper.md`** -- Complete JOSS paper draft (~1,000 words)
2. **`paper/joss/paper.bib`** -- BibTeX references
3. **`paper/softwarex/outline.md`** -- Detailed SoftwareX outline with section content sketches

---

## JOSS Paper Structure

### Title
"happy-simulator: A Discrete-Event Simulation Library for Distributed Systems Engineering"

### Sections (per JOSS template):

1. **Summary** (~100 words) -- What the software does, key capabilities
2. **Statement of Need** (~250 words) -- The gap between analytical queuing theory and full-scale load testing; why existing DES tools (SimPy, Salabim) don't serve software engineers; what happy-simulator provides
3. **Key Design Decisions** (~200 words) -- Generator-based process model (yields = actual delays), nanosecond integer time, uniform target-based dispatch, completion hooks
4. **Component Library** (~150 words) -- Table of component categories (9 queue policies, 5 resilience patterns, 5 rate limiters, 10 cache eviction policies, 6 streaming algorithms, etc.)
5. **Illustrative Example** (~200 words) -- Brief metastable failure scenario showing how simulation reveals non-obvious emergent behavior
6. **Acknowledgements**
7. **References** (~10-15 citations)

---

## SoftwareX Paper Outline

### Section 1: Introduction (~600 words)
- **Hook**: "A single GC pause at 70% utilization caused permanent system collapse in our simulation"
- **Problem**: Engineers can't reason about emergent behavior (feedback loops, amplification cascades) before deployment
- **Gap**: Analytical queuing theory too simplistic; load testing too late; existing DES tools not designed for software engineers
- **Contributions**:
  1. Generator-based DES engine with nanosecond-precision integer time
  2. Component library of 50+ distributed-systems primitives
  3. Case studies demonstrating simulation reveals non-obvious emergent behaviors

### Section 2: Related Work (~600 words)
- SimPy, Salabim, DESMO-J, AnyLogic, Arena (DES tools)
- ns-3, OMNeT++ (network simulators -- too low-level)
- Chaos engineering (tests running systems, not design-time)
- **Differentiation**: Application-level DES with distributed-systems vocabulary

### Section 3: Architecture (~1,200 words)
- 3.1 Core simulation loop (pop-invoke-push) -- `happysimulator/core/simulation.py`
- 3.2 Time representation (integer nanoseconds) -- `happysimulator/core/temporal.py`
- 3.3 Generator-based process model (ProcessContinuation) -- `happysimulator/core/event.py`
- 3.4 Component library taxonomy (table of all categories)
- 3.5 Instrumentation (Probe, Data, TraceRecorder)
- **Figures**: Architecture diagram, SimPy vs happy-simulator code comparison

### Section 4: Case Studies (~1,500 words)
Three case studies, chosen for impact and diversity:

1. **Metastable Failure with Retry Amplification** (`examples/metastable_state.py`)
   - Server at 90% utilization + spike -> queue buildup -> timeouts -> retry storm -> sustained collapse even after load returns to normal
   - Key result: Recovery requires dropping to 50%, far below nominal capacity
   - Figure: Multi-panel (queue depth, goodput, timeout rate, load profile)

2. **GC-Induced Collapse** (`examples/gc_caused_collapse.py`)
   - Single 1s GC pause at 70% utilization -> all in-flight requests timeout -> 2.84x retry amplification -> permanent collapse
   - Controlled comparison: with-retries vs without-retries
   - Figure: Queue depth and goodput comparison

3. **Cache Cold-Start Dynamics** (`examples/cold_start.py`)
   - Zipf-distributed traffic + LRU cache -> top 1% customers get 95%+ hit rate, bottom 50% get <15%
   - Cache reset triggers datastore load spike during recovery
   - Figure: Per-cohort hit rates, datastore load during cold start

### Section 5: Evaluation (~600 words)
- Validation against analytical M/M/1 queue predictions
- Expressiveness: lines of code comparison with SimPy
- Performance: events/second throughput
- Reproducibility: deterministic seeds, CI-tested examples

### Section 6: Discussion (~400 words)
- Limitations: single-threaded, Python performance, alpha status
- Design trade-offs: Python for accessibility, integers for determinism

### Section 7: Future Work (~300 words)
- AI-assisted system design: simulation as a structured sandbox for LLMs
- Natural-language-to-simulation model generation using the component library as vocabulary
- Automated parameter sweep guided by LLM-generated hypotheses
- Framed carefully: "we hypothesize", "preliminary experiments suggest"

### Section 8: Conclusion (~200 words)

---

## Key References to Include

| Ref | Why |
|-----|-----|
| SimPy documentation | Primary comparison point |
| Huang et al., "Metastable Failures in Distributed Systems," OSDI 2022 | Validates metastable failure case study |
| Nichols & Jacobson, "Controlling Queue Delay," ACM Queue 2012 | CoDel queue policy |
| Nygard, "Release It!" | Circuit breaker pattern origin |
| Kleinrock, "Queueing Systems" | Analytical M/M/1 validation |
| Salabim documentation | Related work |
| Banks et al., "Discrete-Event System Simulation" | DES textbook reference |

---

## Implementation Steps

1. Create `paper/joss/` directory structure
2. Write `paper.bib` with all references
3. Write `paper.md` (complete JOSS draft)
4. Create `paper/softwarex/` directory
5. Write `outline.md` (detailed SoftwareX outline with content sketches for each section)
6. Add a brief `paper/README.md` explaining the publication strategy

---

## Verification

- Review JOSS submission requirements at https://joss.readthedocs.io/en/latest/submitting.html
- Ensure `paper.md` follows JOSS Markdown template (YAML front matter with title, authors, affiliations, date, bibliography)
- Verify all referenced examples exist and run (`pytest -q` passes)
- Check that paper.bib entries are valid BibTeX
