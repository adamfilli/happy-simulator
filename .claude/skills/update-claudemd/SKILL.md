---
name: update-claudemd
description: Review recent changes and update CLAUDE.md if needed
---

# Update CLAUDE.md

Review recent git changes and update the project's CLAUDE.md file to reflect any significant changes to the codebase.

## When to Update CLAUDE.md

The CLAUDE.md file should be updated when changes affect:

- **New modules/packages**: New directories or significant files added
- **Public API changes**: New classes, functions, or parameters exposed
- **Architecture changes**: New patterns, abstractions, or workflows
- **Development commands**: New scripts, test commands, or setup steps
- **Dependencies**: New required packages or tools
- **Key examples**: New example files or significant example updates
- **Testing patterns**: New test fixtures, helpers, or conventions
- **Component additions**: New components in `happysimulator/components/`
- **Distribution additions**: New distributions in `happysimulator/distributions/`

## What NOT to Update

Do not update CLAUDE.md for:
- Bug fixes that don't change the API
- Internal refactoring without API changes
- Test additions that follow existing patterns
- Documentation typo fixes
- Minor parameter tweaks

## Instructions

1. **Examine recent changes** using git commands:
   ```bash
   # View recent commits (last 5-10)
   git log --oneline -10

   # View changes since last CLAUDE.md update
   git log --oneline CLAUDE.md

   # View what files changed recently
   git diff --name-only HEAD~5

   # View detailed diff for specific commits
   git show <commit-hash> --stat
   ```

2. **Identify significant changes** by looking for:
   - New files in `happysimulator/` (especially in `core/`, `components/`, `distributions/`)
   - New files in `examples/`
   - Changes to `__init__.py` files (public API changes)
   - New test patterns in `tests/`
   - New development scripts or commands

3. **Read the current CLAUDE.md** to understand existing structure:
   - Quick Reference table
   - Reading Order for New Contributors
   - Development Commands
   - Core Abstractions
   - Architecture
   - Key Directories
   - Testing Patterns
   - Example Patterns
   - Common Patterns
   - Troubleshooting
   - Code Style

4. **Propose updates** by:
   - Identifying which sections need changes
   - Drafting the specific additions/modifications
   - Explaining why each change is needed

5. **Apply updates** to CLAUDE.md:
   - Make minimal, targeted edits
   - Maintain consistent style with existing content
   - Preserve the existing structure
   - Add new entries in appropriate sections
   - **Update the timestamp** at the top of the file (the `> **Last Updated:** YYYY-MM-DD` line)

6. **Verify updates** are correct and complete

## Timestamp Format

CLAUDE.md has a timestamp near the top in this format:

```markdown
> **Last Updated:** 2026-01-31
```

When making ANY updates to CLAUDE.md, update this date to the current date in `YYYY-MM-DD` format.

## Example Output

When running this skill, provide a summary like:

```
## Changes Analyzed
- Commit abc123: Added CircuitBreaker component
- Commit def456: New Pareto distribution

## CLAUDE.md Updates Needed
1. **Key Directories**: Add entry for `components/resilience/`
2. **Common Patterns**: Add CircuitBreaker usage example
3. **Core Abstractions**: Document Pareto distribution

## Updates Applied
- Added CircuitBreaker to components table
- Added ParetoLatency to distributions list
```

## No Updates Needed

If no significant changes warrant CLAUDE.md updates, report:

```
## Changes Analyzed
- Commit abc123: Fixed typo in docstring
- Commit def456: Added unit test for existing feature

## CLAUDE.md Updates Needed
None - recent changes are internal improvements that don't affect the documented API or patterns.
```
