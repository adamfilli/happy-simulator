---
name: release
description: Bump version, commit, push, and create a GitHub release (triggers PyPI publish)
---

# Release

Create a new release: bump the version in `pyproject.toml`, commit, push, and create a GitHub release which triggers the PyPI publish workflow.

## Instructions

1. Read the current version from `pyproject.toml`
2. Parse the version as `major.minor.patch`
3. Ask the user which type of version bump they want using AskUserQuestion:
   - **Patch** (x.y.Z) - Bug fixes, small changes
   - **Minor** (x.Y.0) - New features, backwards compatible
   - **Major** (X.0.0) - Breaking changes
4. Calculate the new version based on their choice
5. Update the version in `pyproject.toml`
6. Report the change (e.g., "Version: 0.2.0 → 0.3.0")
7. Commit with message: `Bump version to {new_version}`
8. Push to the current branch
9. Create a GitHub release:
   ```bash
   gh release create v{new_version} --title "v{new_version}" --generate-notes
   ```
10. Report the release URL to the user

## Notes

- The GitHub release triggers `.github/workflows/publish-pypi.yml` which publishes to PyPI via trusted publishing
- `--generate-notes` auto-generates release notes from PRs and commits since the last tag
- The user must be on `main` (or their release branch) and have `gh` authenticated
