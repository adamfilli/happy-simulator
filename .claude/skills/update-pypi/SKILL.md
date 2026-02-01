---
name: update-pypi
description: Bump the package version for PyPI release
---

# Update PyPI Version

Bump the version in `pyproject.toml` for a new PyPI release.

## Instructions

1. Read the current version from `pyproject.toml`
2. Parse the version as `major.minor.patch`
3. Ask the user which type of version bump they want using AskUserQuestion:
   - **Patch** (x.y.Z) - Bug fixes, small changes
   - **Minor** (x.Y.0) - New features, backwards compatible
   - **Major** (X.0.0) - Breaking changes
4. Calculate the new version based on their choice
5. Update the version in `pyproject.toml`
6. Report the change (e.g., "Updated version: 0.1.3 → 0.1.4")

Note: Publishing to PyPI is handled automatically by GitHub Actions when changes are pushed.
