import { useEffect, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useSimStore } from "./hooks/useSimState";
import { useWebSocket } from "./hooks/useWebSocket";
import GraphView from "./components/GraphView";
import ControlBar from "./components/ControlBar";
import InspectorPanel from "./components/InspectorPanel";
import EventLog from "./components/EventLog";
import type { SimState, Topology, StepResult } from "./types";

export default function App() {
  const { setTopology, setState, addEvents } = useSimStore();
  const { send } = useWebSocket();
  const [activeTab, setActiveTab] = useState<"inspector" | "events">("inspector");

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
      />
      <div className="flex-1 flex overflow-hidden">
        {/* Graph area */}
        <div className="flex-1 relative">
          <ReactFlowProvider>
            <GraphView />
          </ReactFlowProvider>
        </div>

        {/* Right panel */}
        <div className="w-80 border-l border-gray-800 flex flex-col bg-gray-900">
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
          </div>
          <div className="flex-1 overflow-hidden">
            {activeTab === "inspector" ? <InspectorPanel /> : <EventLog />}
          </div>
        </div>
      </div>
    </div>
  );
}
