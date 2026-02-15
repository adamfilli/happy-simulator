import { useEffect, useRef, useCallback } from "react";
import { useSimStore } from "./useSimState";
import type { WSMessage } from "../types";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const {
    setState, addEvents, addEdges, addLogs, setPlaying, setEdgeStats,
    setCodeTrace, setCodePaused, clearCodePaused,
    openCodePanel, closeCodePanel,
    addCodeBreakpoint, removeCodeBreakpoint,
  } = useSimStore();

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const msg: WSMessage = JSON.parse(e.data);

      if (msg.type === "state_update") {
        setState(msg.state);
        if (msg.new_events?.length) addEvents(msg.new_events);
        if (msg.new_edges?.length) addEdges(msg.new_edges);
        if (msg.new_logs?.length) addLogs(msg.new_logs);
        if (msg.edge_stats) setEdgeStats(msg.edge_stats);
        // Process code traces
        if (msg.code_traces?.length) {
          for (const trace of msg.code_traces) {
            setCodeTrace(trace.entity_name, trace);
          }
        }
      } else if (msg.type === "simulation_complete") {
        setPlaying(false);
      } else if (msg.type === "breakpoint_hit") {
        setPlaying(false);
      } else if (msg.type === "code_debug_activated") {
        if (msg.source) {
          openCodePanel(msg.entity_name, {
            entityName: msg.entity_name,
            source: msg.source,
          });
        }
      } else if (msg.type === "code_debug_deactivated") {
        closeCodePanel(msg.entity_name);
      } else if (msg.type === "code_paused") {
        setCodePaused(msg.paused_state);
        setPlaying(false);
      } else if (msg.type === "code_breakpoint_set") {
        addCodeBreakpoint({ id: msg.id, entity_name: msg.entity_name, line_number: msg.line_number });
      } else if (msg.type === "code_breakpoint_removed") {
        if (msg.removed) {
          removeCodeBreakpoint(msg.breakpoint_id);
        }
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
    };
  }, [
    setState, addEvents, addEdges, addLogs, setPlaying, setEdgeStats,
    setCodeTrace, setCodePaused, clearCodePaused,
    openCodePanel, closeCodePanel,
    addCodeBreakpoint, removeCodeBreakpoint,
  ]);

  const send = useCallback((action: string, extra?: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, ...extra }));
    }
  }, []);

  return { send };
}
