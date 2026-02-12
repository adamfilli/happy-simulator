import { create } from "zustand";
import type {
  Topology,
  SimState,
  RecordedEvent,
  RecordedLog,
  TopologyEdge,
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

  setTopology: (t: Topology) => void;
  setState: (s: SimState) => void;
  addEvents: (events: RecordedEvent[]) => void;
  addEdges: (edges: TopologyEdge[]) => void;
  addLogs: (logs: RecordedLog[]) => void;
  setLogLevelFilter: (level: string) => void;
  setPlaying: (p: boolean) => void;
  toggleInternal: () => void;
  selectEntity: (name: string | null) => void;
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
  reset: () => set({ eventLog: [], simLogs: [], isPlaying: false, selectedEntity: null }),
}));
