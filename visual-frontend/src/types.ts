export interface TopologyNode {
  id: string;
  type: string;
  category: string;
}

export interface TopologyEdge {
  source: string;
  target: string;
  kind?: "data" | "probe";
}

export interface Topology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

export interface SimEvent {
  time_s: number;
  event_type: string;
  target: string;
  id: string;
  daemon: boolean;
}

export interface RecordedEvent {
  time_s: number;
  event_type: string;
  target_name: string;
  source_name: string | null;
  event_id: number;
  is_internal: boolean;
}

export interface RecordedLog {
  time_s: number | null;
  wall_time: string;
  level: string;
  logger_name: string;
  message: string;
}

export interface SimState {
  time_s: number;
  events_processed: number;
  heap_size: number;
  is_paused: boolean;
  is_running: boolean;
  is_complete: boolean;
  entities: Record<string, Record<string, unknown>>;
  upcoming: SimEvent[];
}

export interface StepResult {
  state: SimState;
  new_events: RecordedEvent[];
  new_edges: TopologyEdge[];
  new_logs: RecordedLog[];
}

export interface DashboardPanelConfig {
  id: string;
  probeName: string;
  label: string;
  x: number;
  y: number;
}

export type WSMessage =
  | { type: "state_update"; state: SimState; new_events: RecordedEvent[]; new_edges: TopologyEdge[]; new_logs?: RecordedLog[] }
  | { type: "simulation_complete" }
  | { type: "breakpoint_hit" };
