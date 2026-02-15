export interface TopologyNode {
  id: string;
  type: string;
  category: string;
  profile?: { times: number[]; values: number[] };
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
  context?: Record<string, unknown>;
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
  end_time_s?: number | null;
}

export interface EdgeStats {
  [key: string]: { source: string; target: string; count: number; rate: number };
}

export interface StepResult {
  state: SimState;
  new_events: RecordedEvent[];
  new_edges: TopologyEdge[];
  new_logs: RecordedLog[];
  edge_stats?: EdgeStats;
}

export interface ChartConfig {
  chart_id: string;
  title: string;
  y_label: string;
  x_label: string;
  color: string;
  transform: string;
  window_s: number;
  y_min: number | null;
  y_max: number | null;
}

export interface DashboardPanelConfig {
  id: string;
  label: string;
  // react-grid-layout grid units
  x: number;
  y: number;
  w: number;
  h: number;
  // One of these two — probe-based (user-added) or chart-based (predefined)
  probeName?: string;
  chartConfig?: ChartConfig;
}

export interface PinnedChart {
  id: string;
  x: number;
  y: number;
  kind: "entity_metric" | "probe";
  entityName?: string;
  metricKey?: string;
  displayMode?: "total" | "rate" | "avg" | "p99";
  probeName?: string;
  label: string;
}

// --- Code Debug Types ---

export interface CodeTraceRecord {
  line_number: number;
  locals?: Record<string, unknown>;
}

export interface CodeTrace {
  entity_name: string;
  method_name: string;
  start_line: number;
  lines: CodeTraceRecord[];
}

export interface CodePausedState {
  entity_name: string;
  line_number: number;
  locals: Record<string, unknown> | null;
}

export interface EntitySource {
  entity_name: string;
  class_name: string;
  method_name: string;
  source_lines: string[];
  start_line: number;
}

export interface CodeBreakpointInfo {
  id: string;
  entity_name: string;
  line_number: number;
}

export interface CodePanelConfig {
  entityName: string;
  source: EntitySource;
}

export type WSMessage =
  | { type: "state_update"; state: SimState; new_events: RecordedEvent[]; new_edges: TopologyEdge[]; new_logs?: RecordedLog[]; edge_stats?: EdgeStats; code_traces?: CodeTrace[] }
  | { type: "simulation_complete" }
  | { type: "breakpoint_hit" }
  | { type: "code_debug_activated"; entity_name: string; source: EntitySource | null; debug_state: Record<string, unknown> }
  | { type: "code_debug_deactivated"; entity_name: string; debug_state: Record<string, unknown> }
  | { type: "code_paused"; paused_state: CodePausedState }
  | { type: "code_breakpoint_set"; id: string; entity_name: string; line_number: number }
  | { type: "code_breakpoint_removed"; breakpoint_id: string; removed: boolean };
