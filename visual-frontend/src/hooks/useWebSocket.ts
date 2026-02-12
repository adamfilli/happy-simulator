import { useEffect, useRef, useCallback } from "react";
import { useSimStore } from "./useSimState";
import type { WSMessage } from "../types";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const { setState, addEvents, addEdges, setPlaying } = useSimStore();

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
      } else if (msg.type === "simulation_complete") {
        setPlaying(false);
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
    };
  }, [setState, addEvents, addEdges, setPlaying]);

  const send = useCallback((action: string, extra?: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, ...extra }));
    }
  }, []);

  return { send };
}
