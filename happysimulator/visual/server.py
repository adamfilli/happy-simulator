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


def create_app(bridge: SimulationBridge) -> FastAPI:
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
    def get_timeseries(
        probe: str, start_s: float | None = None, end_s: float | None = None
    ) -> JSONResponse:
        return JSONResponse(bridge.get_timeseries(probe, start_s=start_s, end_s=end_s))

    @app.get("/api/charts")
    def get_charts() -> JSONResponse:
        return JSONResponse(bridge.get_chart_configs())

    @app.get("/api/chart_data")
    def get_chart_data(
        chart_id: str, start_s: float | None = None, end_s: float | None = None
    ) -> JSONResponse:
        return JSONResponse(bridge.get_chart_data(chart_id, start_s=start_s, end_s=end_s))

    # --- WebSocket for play mode ---

    @app.websocket("/api/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        play_task: asyncio.Task | None = None
        stop_event = asyncio.Event()

        async def play_loop(speed: int) -> None:
            """Continuously step and push state updates.

            Speed 0 = Max (large batches, minimal delay).
            Speed >0 = wall-clock multiplier (1x/10x/100x).
            """
            import time as _time

            if speed == 0:
                # Max speed: large batches with minimal delay
                while not stop_event.is_set():
                    result = await asyncio.to_thread(bridge.step, 100)
                    try:
                        await ws.send_json({"type": "state_update", **result})
                    except Exception:
                        break
                    if result["state"].get("is_complete"):
                        await ws.send_json({"type": "simulation_complete"})
                        break
                    await asyncio.sleep(0.01)
            else:
                # Time-based: advance sim time proportional to wall clock
                last_wall = _time.monotonic()
                while not stop_event.is_set():
                    now_wall = _time.monotonic()
                    wall_elapsed = now_wall - last_wall
                    target_advance = wall_elapsed * speed
                    last_wall = now_wall
                    current_time = bridge.get_current_time_s()
                    target_time = current_time + target_advance
                    result = await asyncio.to_thread(bridge.run_to, target_time)
                    try:
                        await ws.send_json({"type": "state_update", **result})
                    except Exception:
                        break
                    if result["state"].get("is_complete"):
                        await ws.send_json({"type": "simulation_complete"})
                        break
                    await asyncio.sleep(0.05)

        async def debug_loop(speed: int) -> None:
            """Like play_loop but stops when a breakpoint fires.

            Speed 0 = Max (large batches).
            Speed >0 = wall-clock multiplier.
            """
            import time as _time

            if speed == 0:
                batch_size = 100
            else:
                batch_size = None
            last_wall = _time.monotonic() if speed > 0 else None

            while not stop_event.is_set():
                if batch_size:
                    result = await asyncio.to_thread(bridge.step, batch_size)
                else:
                    now_wall = _time.monotonic()
                    wall_elapsed = now_wall - last_wall  # type: ignore[operator]
                    target_advance = wall_elapsed * speed
                    last_wall = now_wall
                    current_time = bridge.get_current_time_s()
                    target_time = current_time + target_advance
                    result = await asyncio.to_thread(bridge.run_to, target_time)

                try:
                    await ws.send_json({"type": "state_update", **result})
                except Exception:
                    break

                if result["state"].get("is_complete"):
                    await ws.send_json({"type": "simulation_complete"})
                    break

                if result["state"].get("is_paused"):
                    await ws.send_json({"type": "breakpoint_hit"})
                    break

                await asyncio.sleep(0.05 if speed != 0 else 0.01)

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
                    speed = msg.get("speed", 1)
                    play_task = asyncio.create_task(play_loop(speed))

                elif action == "debug":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    stop_event.clear()
                    speed = msg.get("speed", 1)
                    play_task = asyncio.create_task(debug_loop(speed))

                elif action == "pause":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    state = bridge.get_state()
                    await ws.send_json(
                        {
                            "type": "state_update",
                            "state": state,
                            "new_events": [],
                            "new_edges": [],
                            "new_logs": [],
                        }
                    )

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

                elif action == "run_to_event":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    event_number = msg.get("event_number", 0)
                    result = await asyncio.to_thread(bridge.run_to_event, event_number)
                    await ws.send_json({"type": "state_update", **result})

                elif action == "reset":
                    if play_task and not play_task.done():
                        stop_event.set()
                        await play_task
                    state = await asyncio.to_thread(bridge.reset)
                    await ws.send_json(
                        {
                            "type": "state_update",
                            "state": state,
                            "new_events": [],
                            "new_edges": [],
                            "new_logs": [],
                        }
                    )

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
