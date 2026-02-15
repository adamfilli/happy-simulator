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

    @app.get("/api/entity_history")
    def get_entity_history(entity: str) -> JSONResponse:
        return JSONResponse(bridge.get_entity_history(entity))

    @app.get("/api/chart_data")
    def get_chart_data(
        chart_id: str, start_s: float | None = None, end_s: float | None = None
    ) -> JSONResponse:
        return JSONResponse(bridge.get_chart_data(chart_id, start_s=start_s, end_s=end_s))

    # --- Code Debug REST endpoints ---

    @app.get("/api/entity/{name}/source")
    def get_entity_source(name: str) -> JSONResponse:
        source = bridge.get_entity_source(name)
        if source is None:
            return JSONResponse({"error": "Source not found"}, status_code=404)
        return JSONResponse(source)

    @app.post("/api/debug/code/activate")
    def post_activate_code_debug(entity_name: str) -> JSONResponse:
        state = bridge.activate_code_debug(entity_name)
        return JSONResponse(state)

    @app.post("/api/debug/code/deactivate")
    def post_deactivate_code_debug(entity_name: str) -> JSONResponse:
        state = bridge.deactivate_code_debug(entity_name)
        return JSONResponse(state)

    @app.post("/api/debug/code/breakpoints")
    def post_code_breakpoint(entity_name: str, line_number: int) -> JSONResponse:
        result = bridge.set_code_breakpoint(entity_name, line_number)
        return JSONResponse(result)

    @app.delete("/api/debug/code/breakpoints/{bp_id}")
    def delete_code_breakpoint(bp_id: str) -> JSONResponse:
        removed = bridge.remove_code_breakpoint(bp_id)
        return JSONResponse({"removed": removed})

    @app.get("/api/debug/code/state")
    def get_code_debug_state() -> JSONResponse:
        return JSONResponse(bridge.get_code_debug_state())

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

        async def _step_with_code_debug(ws: WebSocket, count: int) -> None:
            """Step that handles code breakpoint blocking.

            When code debugging is active, gen.send() may block the sim thread
            at a code breakpoint.  We run step in a background thread and poll
            for pauses.  While paused, we read WebSocket messages inline so the
            user can send code_continue / code_step / etc. without deadlocking
            the receive loop.

            Returns the set of WS actions consumed so the caller knows not to
            re-process them.
            """
            step_done = asyncio.Event()
            result_holder: list[dict] = []

            async def do_step():
                r = await asyncio.to_thread(bridge.step, count)
                result_holder.append(r)
                step_done.set()

            step_task = asyncio.create_task(do_step())
            pause_notified = False

            while not step_done.is_set():
                # Check if sim thread is paused at a code breakpoint
                if bridge._code_debugger.is_code_paused():
                    if not pause_notified:
                        paused_state = bridge._code_debugger.get_paused_state()
                        try:
                            await ws.send_json({"type": "code_paused", "paused_state": paused_state})
                        except Exception:
                            break
                        pause_notified = True

                    # While paused, read WS messages so user can resume
                    try:
                        raw = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                        msg = json.loads(raw)
                        action = msg.get("action")
                        if action == "code_continue":
                            bridge.code_continue()
                            pause_notified = False
                        elif action == "code_step":
                            bridge.code_step()
                            pause_notified = False
                        elif action == "code_step_over":
                            bridge.code_step_over()
                            pause_notified = False
                        elif action == "code_step_out":
                            bridge.code_step_out()
                            pause_notified = False
                        # Ignore other actions while paused
                    except asyncio.TimeoutError:
                        pass
                else:
                    pause_notified = False
                    await asyncio.sleep(0.05)

            await step_task
            if result_holder:
                try:
                    await ws.send_json({"type": "state_update", **result_holder[0]})
                except Exception:
                    pass

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
                    has_active = len(bridge._code_debugger._active_entities) > 0
                    if has_active:
                        await _step_with_code_debug(ws, count)
                    else:
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

                # --- Code Debug WS actions ---

                elif action == "activate_code_debug":
                    entity_name = msg.get("entity_name", "")
                    state = await asyncio.to_thread(bridge.activate_code_debug, entity_name)
                    source = await asyncio.to_thread(bridge.get_entity_source, entity_name)
                    await ws.send_json({"type": "code_debug_activated", "entity_name": entity_name, "source": source, "debug_state": state})

                elif action == "deactivate_code_debug":
                    entity_name = msg.get("entity_name", "")
                    state = await asyncio.to_thread(bridge.deactivate_code_debug, entity_name)
                    await ws.send_json({"type": "code_debug_deactivated", "entity_name": entity_name, "debug_state": state})

                elif action == "set_code_breakpoint":
                    entity_name = msg.get("entity_name", "")
                    line_number = msg.get("line_number", 0)
                    result = await asyncio.to_thread(bridge.set_code_breakpoint, entity_name, line_number)
                    await ws.send_json({"type": "code_breakpoint_set", **result})

                elif action == "remove_code_breakpoint":
                    bp_id = msg.get("breakpoint_id", "")
                    removed = await asyncio.to_thread(bridge.remove_code_breakpoint, bp_id)
                    await ws.send_json({"type": "code_breakpoint_removed", "breakpoint_id": bp_id, "removed": removed})

                elif action == "code_continue":
                    bridge.code_continue()

                elif action == "code_step":
                    bridge.code_step()

                elif action == "code_step_over":
                    bridge.code_step_over()

                elif action == "code_step_out":
                    bridge.code_step_out()

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
