---
name: git-back-to-main
description: Discard local changes, switch to main, and pull latest
---

# Git Back to Main

Abandon the current branch and return to a clean, up-to-date `main`.

## Instructions

1. Record the current branch name with `git branch --show-current`.
2. Discard all local changes (staged, unstaged, and untracked):
   ```
   git checkout -- .
   git clean -fd
   ```
3. Switch to main: `git checkout main`
4. Pull latest: `git pull origin main`
5. If the pull fails (e.g. diverged history), hard reset to origin:
   ```
   git fetch origin main
   git reset --hard origin/main
   ```
6. Report: previous branch name, current branch, and HEAD commit.
