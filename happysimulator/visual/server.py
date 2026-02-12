"""FastAPI application for the visual debugger.

Provides REST endpoints and a WebSocket for browser communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from happysimulator.visual.bridge import SimulationBridge

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(bridge: "SimulationBridge") -> FastAPI:
    """Create the FastAPI application wired to the given bridge."""
    app = FastAPI(title="Happy Simulator Debugger")

    # --- REST endpoints ---

    @app.get("/api/topology")
    def get_topology() -> JSONResponse:
        return JSONResponse(bridge.get_topology())

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        return JSONResponse(bridge.get_state())

    @app.post("/api/step")
    def post_step(count: int = 1) -> JSONResponse:
        result = bridge.step(count)
        return JSONResponse(result)

    @app.post("/api/reset")
    def post_reset() -> JSONResponse:
        state = bridge.reset()
        return JSONResponse(state)

    @app.post("/api/run_to")
    def post_run_to(time_s: float) -> JSONResponse:
        result = bridge.run_to(time_s)
        return JSONResponse(result)

    @app.get("/api/events")
    def get_events(last_n: int = 100) -> JSONResponse:
        return JSONResponse(bridge.get_event_log(last_n))

    @app.get("/api/probes")
    def get_probes() -> JSONResponse:
        return JSONResponse(bridge.list_probes())

    @app.get("/api/timeseries")
    def get_timeseries(probe: str) -> JSONResponse:
        return JSONResponse(bridge.get_timeseries(probe))

    # --- WebSocket for play mode ---

    @app.websocket("/api/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        play_task: asyncio.Task | None = None
        stop_event = asyncio.Event()

        async def play_loop(speed: int) -> None:
            """Continuously step and push state updates."""
            while not stop_event.is_set():
                result = await asyncio.to_thread(bridge.step, speed)
                try:
                    await ws.send_json({"type": "state_update", **result})
                except Exception:
                    break

                # Check if simulation completed
                if result["state"].get("is_complete"):
                    await ws.send_json({"type": "simulation_complete"})
                    break

                # Small delay to avoid flooding the browser
                await asyncio.sleep(0.05)

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                action = msg.get("action")

                if action == "play":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    stop_event.clear()
                    speed = msg.get("speed", 10)
                    play_task = asyncio.create_task(play_loop(speed))

                elif action == "pause":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    state = bridge.get_state()
                    await ws.send_json({"type": "state_update", "state": state, "new_events": [], "new_edges": []})

                elif action == "step":
                    count = msg.get("count", 1)
                    result = await asyncio.to_thread(bridge.step, count)
                    await ws.send_json({"type": "state_update", **result})

                elif action == "run_to":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    time_s = msg.get("time_s", 0)
                    result = await asyncio.to_thread(bridge.run_to, time_s)
                    await ws.send_json({"type": "state_update", **result})

                elif action == "reset":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    state = await asyncio.to_thread(bridge.reset)
                    await ws.send_json({"type": "state_update", "state": state, "new_events": [], "new_edges": []})

        except WebSocketDisconnect:
            if play_task and not play_task.done():
                stop_event.set()

    # --- Static files ---

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    return app
