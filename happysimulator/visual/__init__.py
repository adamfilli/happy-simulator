"""Browser-based simulation debugger.

Usage::

    from happysimulator.visual import serve, Chart

    sim = Simulation(sources=[source], entities=[server, sink])
    serve(sim)  # opens browser, step through interactively
"""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from happysimulator.visual.dashboard import Chart

if TYPE_CHECKING:
    from happysimulator.core.simulation import Simulation

__all__ = ["Chart", "serve"]


def serve(
    sim: Simulation,
    *,
    charts: list[Chart] | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Launch the simulation debugger in a browser.

    Pauses the simulation at t=0, starts a local web server, and opens
    the debugger UI.  Blocks until the server is stopped (Ctrl+C).

    Args:
        sim: A simulation that has NOT been run yet.
        charts: Optional list of predefined Chart objects to show on the Dashboard.
        host: Bind address for the web server.
        port: Port for the web server.
        open_browser: Whether to open the default browser automatically.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "The visual debugger requires extra dependencies.\n"
            "Install them with:  pip install happysim[visual]"
        ) from None

    from happysimulator.visual.bridge import SimulationBridge
    from happysimulator.visual.server import create_app

    if sim._is_running:
        raise RuntimeError("Cannot serve a simulation that is already running.")

    bridge = SimulationBridge(sim, charts=charts or [])
    app = create_app(bridge)

    # Pause at t=0 then prime the heap
    sim.control.pause()
    sim.run()

    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        bridge.close()
