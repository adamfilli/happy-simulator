# Publication Strategy

This directory contains academic paper materials for `happy-simulator`.

## Two-Stage Approach

### Stage 1: JOSS (Journal of Open Source Software) — Start here

- **Directory**: `joss/`
- **Files**: `paper.md` (complete draft), `paper.bib` (references)
- **Format**: ~1,000 words, short software description
- **Review**: Transparent GitHub-based review focused on software quality
- **Timeline**: 2–3 weeks polish, 4–8 weeks review ≈ 3 months to publication
- **Why first**: Fastest path to a citable DOI. The project already meets JOSS
  requirements (installable, tested, documented, Apache 2.0 licensed, examples).

### Stage 2: SoftwareX (Elsevier) — Expand later

- **Directory**: `softwarex/`
- **Files**: `outline.md` (detailed outline with content sketches)
- **Format**: 3,000–6,000 words with case studies and evaluation
- **Timeline**: 4–6 weeks writing, 3–6 months review
- **Why second**: Deeper technical presentation of architecture and case studies.
  Can reference the JOSS DOI. No conflict—JOSS is a software description,
  SoftwareX is a research paper.

## Before Submitting to JOSS

1. Add your ORCID to `joss/paper.md` front matter (replace `0000-0000-0000-0000`)
2. Review and polish the draft
3. Ensure `pytest -q` passes (all examples are CI-tested)
4. Verify the repository meets JOSS requirements:
   - [x] Open source license (Apache 2.0)
   - [x] Installable (`pip install -e .`)
   - [x] Automated tests (`pytest`)
   - [x] Documentation (CLAUDE.md, docstrings, examples)
   - [x] Examples with expected output
   - [ ] 6+ months of public history with issues/PRs
5. Submit at https://joss.theoj.org/papers/new
