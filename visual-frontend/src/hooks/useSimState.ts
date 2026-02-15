import { create } from "zustand";
import type {
  Topology,
  SimState,
  RecordedEvent,
  RecordedLog,
  TopologyEdge,
  DashboardPanelConfig,
  EdgeStats,
  CodePanelConfig,
  CodeTrace,
  CodePausedState,
  CodeBreakpointInfo,
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
  edgeStats: EdgeStats;

  // Code debug state
  codePanels: Map<string, CodePanelConfig>;
  codeTraces: Map<string, CodeTrace>;
  codePausedEntity: string | null;
  codePausedState: CodePausedState | null;
  codeBreakpoints: CodeBreakpointInfo[];

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
  setEdgeStats: (stats: EdgeStats) => void;

  // Code debug actions
  openCodePanel: (entityName: string, config: CodePanelConfig) => void;
  closeCodePanel: (entityName: string) => void;
  setCodeTrace: (entityName: string, trace: CodeTrace) => void;
  setCodePaused: (state: CodePausedState | null) => void;
  clearCodePaused: () => void;
  addCodeBreakpoint: (bp: CodeBreakpointInfo) => void;
  removeCodeBreakpoint: (bpId: string) => void;

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
  edgeStats: {},

  // Code debug state
  codePanels: new Map(),
  codeTraces: new Map(),
  codePausedEntity: null,
  codePausedState: null,
  codeBreakpoints: [],

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
  setEdgeStats: (stats) => set({ edgeStats: stats }),

  // Code debug actions
  openCodePanel: (entityName, config) =>
    set((prev) => {
      const panels = new Map(prev.codePanels);
      panels.set(entityName, config);
      return { codePanels: panels };
    }),
  closeCodePanel: (entityName) =>
    set((prev) => {
      const panels = new Map(prev.codePanels);
      panels.delete(entityName);
      const traces = new Map(prev.codeTraces);
      traces.delete(entityName);
      return { codePanels: panels, codeTraces: traces };
    }),
  setCodeTrace: (entityName, trace) =>
    set((prev) => {
      const traces = new Map(prev.codeTraces);
      traces.set(entityName, trace);
      return { codeTraces: traces };
    }),
  setCodePaused: (state) =>
    set({ codePausedEntity: state?.entity_name ?? null, codePausedState: state }),
  clearCodePaused: () =>
    set({ codePausedEntity: null, codePausedState: null }),
  addCodeBreakpoint: (bp) =>
    set((prev) => ({ codeBreakpoints: [...prev.codeBreakpoints, bp] })),
  removeCodeBreakpoint: (bpId) =>
    set((prev) => ({ codeBreakpoints: prev.codeBreakpoints.filter((b) => b.id !== bpId) })),

  reset: () => set({
    eventLog: [], simLogs: [], isPlaying: false, selectedEntity: null,
    dashboardTimeRange: { start: null, end: null }, edgeStats: {},
    codePanels: new Map(), codeTraces: new Map(),
    codePausedEntity: null, codePausedState: null, codeBreakpoints: [],
  }),
}));
