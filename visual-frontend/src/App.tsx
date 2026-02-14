import { useEffect, useState, useRef, useCallback } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useSimStore } from "./hooks/useSimState";
import { useWebSocket } from "./hooks/useWebSocket";
import GraphView from "./components/GraphView";
import DashboardView from "./components/DashboardView";
import ControlBar from "./components/ControlBar";
import Timeline from "./components/Timeline";
import BreakpointPanel from "./components/BreakpointPanel";
import InspectorPanel from "./components/InspectorPanel";
import EventLog from "./components/EventLog";
import SimulationLog from "./components/SimulationLog";
import type { SimState, Topology, StepResult, ChartConfig, BreakpointInfo } from "./types";

export default function App() {
  const { setTopology, setState, addEvents, addLogs, addDashboardPanel, setEdgeStats } = useSimStore();
  const activeView = useSimStore((s) => s.activeView);
  const setActiveView = useSimStore((s) => s.setActiveView);
  const state = useSimStore((s) => s.state);
  const topology = useSimStore((s) => s.topology);
  const { send } = useWebSocket();
  const [activeTab, setActiveTab] = useState<"inspector" | "events" | "logs">("inspector");
  const [panelWidth, setPanelWidth] = useState(320);
  const dragging = useRef(false);
  const [bpPanelOpen, setBpPanelOpen] = useState(false);
  const [breakpointCount, setBreakpointCount] = useState(0);

  const entityNames = (topology?.nodes ?? []).map((n) => n.id);

  // Fetch breakpoint count periodically
  useEffect(() => {
    const fetchBpCount = () => {
      fetch("/api/breakpoints")
        .then((r) => r.json())
        .then((bps: BreakpointInfo[]) => setBreakpointCount(bps.length))
        .catch(() => {});
    };
    fetchBpCount();
    const interval = setInterval(fetchBpCount, 2000);
    return () => clearInterval(interval);
  }, []);

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

  // Fetch initial topology + state + predefined charts
  useEffect(() => {
    Promise.all([
      fetch("/api/topology").then((r) => r.json() as Promise<Topology>),
      fetch("/api/state").then((r) => r.json() as Promise<SimState>),
      fetch("/api/charts").then((r) => r.json() as Promise<ChartConfig[]>).catch(() => [] as ChartConfig[]),
    ]).then(([topo, initialState, charts]) => {
      setTopology(topo);
      setState(initialState);
      if (charts.length > 0) {
        for (let i = 0; i < charts.length; i++) {
          const c = charts[i];
          addDashboardPanel({
            id: c.chart_id,
            label: c.title || `Chart ${c.chart_id}`,
            chartConfig: c,
            x: (i % 3) * 4,
            y: Math.floor(i / 3) * 4,
            w: 4,
            h: 4,
          });
        }
        setActiveView("dashboard");
      }
    });
  }, [setTopology, setState, addDashboardPanel, setActiveView]);

  const applyStepResult = (data: StepResult) => {
    setState(data.state);
    if (data.new_events?.length) addEvents(data.new_events);
    if (data.new_logs?.length) addLogs(data.new_logs);
    if (data.edge_stats) setEdgeStats(data.edge_stats);
  };

  const handleStep = async (count: number) => {
    const res = await fetch(`/api/step?count=${count}`, { method: "POST" });
    const data: StepResult = await res.json();
    applyStepResult(data);
  };

  const handlePlay = (speed: number) => {
    send("play", { speed });
  };

  const handleDebug = (speed: number) => {
    send("debug", { speed });
  };

  const handlePause = () => {
    send("pause");
  };

  const handleRunTo = async (time_s: number) => {
    const res = await fetch(`/api/run_to?time_s=${time_s}`, { method: "POST" });
    const data: StepResult = await res.json();
    applyStepResult(data);
  };

  const handleRunToEvent = (eventNumber: number) => {
    send("run_to_event", { event_number: eventNumber });
  };

  const handleReset = async () => {
    useSimStore.getState().reset();
    const res = await fetch("/api/reset", { method: "POST" });
    const resetState: SimState = await res.json();
    setState(resetState);
    // Re-fetch topology after reset
    const topoRes = await fetch("/api/topology");
    const topo: Topology = await topoRes.json();
    setTopology(topo);
  };

  const handleSeekTo = (time_s: number) => {
    handleRunTo(time_s);
  };

  // Gather time-based breakpoints for timeline markers
  const [timeBreakpoints, setTimeBreakpoints] = useState<Array<{ id: string; type: string; time_s?: number }>>([]);
  useEffect(() => {
    if (bpPanelOpen) {
      fetch("/api/breakpoints")
        .then((r) => r.json())
        .then((bps: BreakpointInfo[]) => {
          setTimeBreakpoints(bps.filter((bp) => bp.time_s != null));
          setBreakpointCount(bps.length);
        })
        .catch(() => {});
    }
  }, [bpPanelOpen, state?.events_processed]);

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      <ControlBar
        onStep={handleStep}
        onPlay={handlePlay}
        onDebug={handleDebug}
        onPause={handlePause}
        onReset={handleReset}
        onRunTo={handleRunTo}
        onRunToEvent={handleRunToEvent}
        onToggleBreakpoints={() => setBpPanelOpen((v) => !v)}
        breakpointCount={breakpointCount}
      />
      {bpPanelOpen && (
        <BreakpointPanel
          open={bpPanelOpen}
          onClose={() => setBpPanelOpen(false)}
          entityNames={entityNames}
        />
      )}
      <Timeline
        currentTime={state?.time_s ?? 0}
        endTime={state?.end_time_s ?? null}
        breakpoints={timeBreakpoints}
        onSeekTo={handleSeekTo}
      />
      <div className="flex-1 flex overflow-hidden">
        {/* Main content area */}
        <div className="flex-1 flex flex-col relative overflow-hidden">
          {/* View toggle tabs */}
          <div className="flex border-b border-gray-800 bg-gray-900 shrink-0">
            <button
              onClick={() => setActiveView("graph")}
              className={`px-4 py-1.5 text-xs font-medium ${
                activeView === "graph"
                  ? "text-white border-b-2 border-blue-500"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              Graph
            </button>
            <button
              onClick={() => setActiveView("dashboard")}
              className={`px-4 py-1.5 text-xs font-medium ${
                activeView === "dashboard"
                  ? "text-white border-b-2 border-blue-500"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              Dashboard
            </button>
          </div>
          {/* View content */}
          <div className="flex-1 relative overflow-hidden">
            {activeView === "graph" ? (
              <ReactFlowProvider>
                <GraphView />
              </ReactFlowProvider>
            ) : (
              <DashboardView />
            )}
          </div>
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
