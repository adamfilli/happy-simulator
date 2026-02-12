import { useEffect, useState, useRef, useCallback } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useSimStore } from "./hooks/useSimState";
import { useWebSocket } from "./hooks/useWebSocket";
import GraphView from "./components/GraphView";
import ControlBar from "./components/ControlBar";
import InspectorPanel from "./components/InspectorPanel";
import EventLog from "./components/EventLog";
import SimulationLog from "./components/SimulationLog";
import type { SimState, Topology, StepResult } from "./types";

export default function App() {
  const { setTopology, setState, addEvents, addLogs } = useSimStore();
  const { send } = useWebSocket();
  const [activeTab, setActiveTab] = useState<"inspector" | "events" | "logs">("inspector");
  const [panelWidth, setPanelWidth] = useState(320);
  const dragging = useRef(false);

  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;

    const onMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const newWidth = window.innerWidth - ev.clientX;
      setPanelWidth(Math.max(240, Math.min(newWidth, window.innerWidth * 0.7)));
    };
    const onUp = () => {
      dragging.current = false;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);

  // Fetch initial topology + state
  useEffect(() => {
    Promise.all([
      fetch("/api/topology").then((r) => r.json() as Promise<Topology>),
      fetch("/api/state").then((r) => r.json() as Promise<SimState>),
    ]).then(([topo, state]) => {
      setTopology(topo);
      setState(state);
    });
  }, [setTopology, setState]);

  const handleStep = async (count: number) => {
    const res = await fetch(`/api/step?count=${count}`, { method: "POST" });
    const data: StepResult = await res.json();
    setState(data.state);
    if (data.new_events?.length) addEvents(data.new_events);
    if (data.new_logs?.length) addLogs(data.new_logs);
  };

  const handlePlay = (speed: number) => {
    send("play", { speed });
  };

  const handlePause = () => {
    send("pause");
  };

  const handleRunTo = async (time_s: number) => {
    const res = await fetch(`/api/run_to?time_s=${time_s}`, { method: "POST" });
    const data: StepResult = await res.json();
    setState(data.state);
    if (data.new_events?.length) addEvents(data.new_events);
    if (data.new_logs?.length) addLogs(data.new_logs);
  };

  const handleRunToEvent = (eventNumber: number) => {
    send("run_to_event", { event_number: eventNumber });
  };

  const handleReset = async () => {
    useSimStore.getState().reset();
    const res = await fetch("/api/reset", { method: "POST" });
    const state: SimState = await res.json();
    setState(state);
    // Re-fetch topology after reset
    const topoRes = await fetch("/api/topology");
    const topo: Topology = await topoRes.json();
    setTopology(topo);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      <ControlBar
        onStep={handleStep}
        onPlay={handlePlay}
        onPause={handlePause}
        onReset={handleReset}
        onRunTo={handleRunTo}
        onRunToEvent={handleRunToEvent}
      />
      <div className="flex-1 flex overflow-hidden">
        {/* Graph area */}
        <div className="flex-1 relative">
          <ReactFlowProvider>
            <GraphView />
          </ReactFlowProvider>
        </div>

        {/* Resize handle */}
        <div
          onMouseDown={onDragStart}
          className="w-1 cursor-col-resize bg-gray-800 hover:bg-blue-500 active:bg-blue-500 transition-colors shrink-0"
        />

        {/* Right panel */}
        <div style={{ width: panelWidth }} className="flex flex-col bg-gray-900 shrink-0">
          <div className="flex border-b border-gray-800">
            <button
              onClick={() => setActiveTab("inspector")}
              className={`flex-1 px-3 py-2 text-xs font-medium ${
                activeTab === "inspector"
                  ? "text-white border-b-2 border-blue-500"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              Inspector
            </button>
            <button
              onClick={() => setActiveTab("events")}
              className={`flex-1 px-3 py-2 text-xs font-medium ${
                activeTab === "events"
                  ? "text-white border-b-2 border-blue-500"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              Event Log
            </button>
            <button
              onClick={() => setActiveTab("logs")}
              className={`flex-1 px-3 py-2 text-xs font-medium ${
                activeTab === "logs"
                  ? "text-white border-b-2 border-blue-500"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              Sim Log
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            {activeTab === "inspector" ? <InspectorPanel /> : activeTab === "events" ? <EventLog /> : <SimulationLog />}
          </div>
        </div>
      </div>
    </div>
  );
}
