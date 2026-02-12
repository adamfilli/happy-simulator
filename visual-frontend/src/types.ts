export interface TopologyNode {
  id: string;
  type: string;
  category: string;
}

export interface TopologyEdge {
  source: string;
  target: string;
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
  event_id: string;
  is_internal: boolean;
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
}

export type WSMessage =
  | { type: "state_update"; state: SimState; new_events: RecordedEvent[]; new_edges: TopologyEdge[] }
  | { type: "simulation_complete" };
