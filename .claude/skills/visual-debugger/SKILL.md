---
name: visual-debugger
description: Launch the browser-based simulation debugger
disable-model-invocation: true
---

# Visual Debugger

Launch the browser-based simulation debugger at http://127.0.0.1:8765.

## Instructions

1. Run: `python examples/visual_debugger.py`
2. This starts the debugger server in the background and opens the browser automatically
3. Tell the user the server is running at http://127.0.0.1:8765
4. Remind them to press Ctrl+C in the terminal to stop the server when done

## Rebuild After Frontend Changes

After modifying any file under `visual-frontend/src/`, always rebuild so the static assets served by the Python backend are up to date:

```bash
cd visual-frontend && npm run build
```

This runs `tsc -b && vite build`, outputting to `happysimulator/visual/static/`. The built assets (hashed `.js` and `.css` files) should be committed alongside the source changes.

## Clean Build & Reset

If the debugger is showing stale UI that doesn't reflect recent frontend changes, do a full clean rebuild:

1. Stop any running `serve()` processes
2. Delete built output: `rm -rf happysimulator/visual/static/*`
3. Delete TypeScript incremental cache: `rm visual-frontend/tsconfig.tsbuildinfo`
4. Rebuild from scratch: `cd visual-frontend && npm run build`
5. Restart the debugger: `python examples/visual_debugger.py`
6. Hard-refresh the browser with **Ctrl+Shift+R** (or open an incognito window) to bypass browser cache
