import { create } from "zustand";
import type {
  Topology,
  SimState,
  RecordedEvent,
  RecordedLog,
  TopologyEdge,
  DashboardPanelConfig,
} from "../types";

interface SimStore {
  topology: Topology | null;
  state: SimState | null;
  eventLog: RecordedEvent[];
  simLogs: RecordedLog[];
  logLevelFilter: string;
  isPlaying: boolean;
  showInternal: boolean;
  selectedEntity: string | null;
  dashboardPanels: DashboardPanelConfig[];
  activeView: "graph" | "dashboard";
  dashboardTimeRange: { start: number | null; end: number | null };

  setTopology: (t: Topology) => void;
  setState: (s: SimState) => void;
  addEvents: (events: RecordedEvent[]) => void;
  addEdges: (edges: TopologyEdge[]) => void;
  addLogs: (logs: RecordedLog[]) => void;
  setLogLevelFilter: (level: string) => void;
  setPlaying: (p: boolean) => void;
  toggleInternal: () => void;
  selectEntity: (name: string | null) => void;
  addDashboardPanel: (panel: DashboardPanelConfig) => void;
  removeDashboardPanel: (id: string) => void;
  updateDashboardLayout: (layout: Array<{ i: string; x: number; y: number; w: number; h: number }>) => void;
  setActiveView: (view: "graph" | "dashboard") => void;
  setDashboardTimeRange: (range: { start: number | null; end: number | null }) => void;
  reset: () => void;
}

const MAX_EVENT_LOG = 2000;
const MAX_SIM_LOGS = 2000;

export const useSimStore = create<SimStore>((set) => ({
  topology: null,
  state: null,
  eventLog: [],
  simLogs: [],
  logLevelFilter: "DEBUG",
  isPlaying: false,
  showInternal: false,
  selectedEntity: null,
  dashboardPanels: [],
  activeView: "graph",
  dashboardTimeRange: { start: null, end: null },

  setTopology: (t) => set({ topology: t }),
  setState: (s) => set({ state: s }),
  addEvents: (events) =>
    set((prev) => {
      const merged = [...prev.eventLog, ...events];
      return { eventLog: merged.slice(-MAX_EVENT_LOG) };
    }),
  addEdges: (edges) =>
    set((prev) => {
      if (!prev.topology || edges.length === 0) return {};
      return {
        topology: {
          ...prev.topology,
          edges: [...prev.topology.edges, ...edges],
        },
      };
    }),
  addLogs: (logs) =>
    set((prev) => {
      const merged = [...prev.simLogs, ...logs];
      return { simLogs: merged.slice(-MAX_SIM_LOGS) };
    }),
  setLogLevelFilter: (level) => set({ logLevelFilter: level }),
  setPlaying: (p) => set({ isPlaying: p }),
  toggleInternal: () => set((prev) => ({ showInternal: !prev.showInternal })),
  selectEntity: (name) => set({ selectedEntity: name }),
  addDashboardPanel: (panel) =>
    set((prev) => ({ dashboardPanels: [...prev.dashboardPanels, panel] })),
  removeDashboardPanel: (id) =>
    set((prev) => ({ dashboardPanels: prev.dashboardPanels.filter((p) => p.id !== id) })),
  updateDashboardLayout: (layout) =>
    set((prev) => ({
      dashboardPanels: prev.dashboardPanels.map((p) => {
        const item = layout.find((l) => l.i === p.id);
        return item ? { ...p, x: item.x, y: item.y, w: item.w, h: item.h } : p;
      }),
    })),
  setActiveView: (view) => set({ activeView: view }),
  setDashboardTimeRange: (range) => set({ dashboardTimeRange: range }),
  reset: () => set({ eventLog: [], simLogs: [], isPlaying: false, selectedEntity: null, dashboardTimeRange: { start: null, end: null } }),
}));
