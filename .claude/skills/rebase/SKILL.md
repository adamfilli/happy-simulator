---
name: rebase
description: Pull latest main and rebase the current branch on top of it
---

# Rebase on Main

Pull the latest `main` branch from origin and rebase the current branch on top of it.

## Instructions

1. Verify you are NOT on `main`. If on `main`, inform the user and stop.
2. Fetch the latest from origin: `git fetch origin main`
3. Rebase the current branch onto the updated main: `git rebase origin/main`
4. If the rebase succeeds, report the number of commits replayed and the current HEAD
5. If there are conflicts:
   - List the conflicting files
   - Show the conflict markers in each file
   - Ask the user how they want to resolve each conflict
   - Do NOT abort the rebase without user confirmation
6. After completion, run `git log --oneline origin/main..HEAD` to show the rebased commits
