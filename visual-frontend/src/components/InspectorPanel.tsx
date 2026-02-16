import { useCallback, useEffect, useState, useRef } from "react";
import { useSimStore } from "../hooks/useSimState";
import TimeSeriesChart from "./TimeSeriesChart";
import MiniSparkline from "./MiniSparkline";
import { toRate, toBucketed, type Series } from "./sparklineTransforms";
import type { GroupMembersResponse, GroupMember } from "../types";

type MetricMode = "total" | "rate" | "avg" | "p99";
const MODE_CYCLE: MetricMode[] = ["total", "rate", "avg", "p99"];
const MODE_LABELS: Record<MetricMode, string> = {
  total: "total",
  rate: "/s",
  avg: "avg",
  p99: "p99",
};

const DRAG_MIME = "application/happysim-chart";

interface TimeSeriesData {
  name: string;
  metric: string;
  target: string;
  times: number[];
  values: number[];
}

interface EntityHistoryData {
  entity: string;
  metrics: Record<string, { times: number[]; values: number[] }>;
}

const CATEGORY_COLORS: Record<string, string> = {
  source: "#22c55e",
  queued_resource: "#3b82f6",
  sink: "#ef4444",
  rate_limiter: "#f97316",
  router: "#a855f7",
  resource: "#a16207",
  probe: "#06b6d4",
  other: "#6b7280",
};

// --- Group Inspector Sub-component ---

function GroupInspector({
  groupId,
  nodeInfo,
}: {
  groupId: string;
  nodeInfo: { type: string; category: string; member_count?: number };
}) {
  const extractEntity = useSimStore((s) => s.extractEntity);
  const [members, setMembers] = useState<GroupMember[]>([]);
  const [total, setTotal] = useState(nodeInfo.member_count ?? 0);
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [expandedMember, setExpandedMember] = useState<string | null>(null);
  const [expandedState, setExpandedState] = useState<Record<string, unknown> | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const color = CATEGORY_COLORS[nodeInfo.category] || CATEGORY_COLORS.other;

  const fetchMembers = useCallback(
    (searchTerm: string, fetchOffset: number, append: boolean) => {
      const params = new URLSearchParams({
        group_id: groupId,
        offset: String(fetchOffset),
        limit: "50",
      });
      if (searchTerm) params.set("search", searchTerm);
      fetch(`/api/group_members?${params}`)
        .then((r) => r.json())
        .then((data: GroupMembersResponse) => {
          setMembers((prev) => (append ? [...prev, ...data.members] : data.members));
          setTotal(data.total);
        });
    },
    [groupId]
  );

  // Initial fetch and search changes
  useEffect(() => {
    setOffset(0);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchMembers(search, 0, false);
    }, 200);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [search, fetchMembers]);

  const loadMore = () => {
    const newOffset = offset + 50;
    setOffset(newOffset);
    fetchMembers(search, newOffset, true);
  };

  const handleExpandMember = (name: string) => {
    if (expandedMember === name) {
      setExpandedMember(null);
      setExpandedState(null);
      return;
    }
    setExpandedMember(name);
    fetch(`/api/entity_state?entity=${encodeURIComponent(name)}`)
      .then((r) => r.json())
      .then((data: { entity: string; state: Record<string, unknown> }) => {
        setExpandedState(data.state);
      });
  };

  return (
    <div className="p-3 space-y-3">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-white">{nodeInfo.type}</h3>
          <span
            className="px-1.5 py-0.5 text-[10px] font-bold rounded-full"
            style={{ backgroundColor: `${color}25`, color }}
          >
            {total.toLocaleString()}
          </span>
        </div>
        <span className="text-xs text-gray-500">Entity Group</span>
      </div>

      {/* Search bar */}
      <input
        type="text"
        placeholder="Search members..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full px-2 py-1 text-xs bg-gray-900 border border-gray-700 rounded text-white placeholder-gray-600 focus:outline-none focus:border-gray-500"
      />

      {/* Member list */}
      <div className="space-y-0.5 max-h-[400px] overflow-y-auto">
        {members.map((member) => {
          const isExpanded = expandedMember === member.name;
          // Pick top 2 metrics for the row summary
          const topMetrics = Object.entries(member.state)
            .filter(([, v]) => typeof v === "number")
            .slice(0, 2);

          return (
            <div key={member.name}>
              <div
                className="flex items-center justify-between px-2 py-1.5 rounded hover:bg-gray-800/50 cursor-pointer group"
                onClick={() => handleExpandMember(member.name)}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-xs text-white truncate">{member.name}</span>
                  {topMetrics.map(([k, v]) => (
                    <span key={k} className="text-[10px] text-gray-500 font-mono">
                      {k}={typeof v === "number" && !Number.isInteger(v) ? (v as number).toFixed(2) : String(v)}
                    </span>
                  ))}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    extractEntity(member.name, groupId);
                  }}
                  className="hidden group-hover:block text-[10px] text-blue-400 hover:text-blue-300 px-1.5 py-0.5 rounded border border-gray-700 hover:border-gray-600 whitespace-nowrap"
                >
                  Extract
                </button>
              </div>

              {/* Expanded member detail */}
              {isExpanded && expandedState && (
                <div className="ml-4 px-2 py-1.5 bg-gray-900/50 rounded mb-1 space-y-0.5">
                  {Object.entries(expandedState).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-[10px]">
                      <span className="text-gray-400">{key}</span>
                      <span className="text-white font-mono">{formatValue(value)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Load more */}
      {members.length < total && (
        <button
          onClick={loadMore}
          className="w-full px-2 py-1 text-xs text-gray-400 hover:text-gray-300 hover:bg-gray-800 rounded border border-gray-700"
        >
          Load more ({members.length} / {total.toLocaleString()})
        </button>
      )}
    </div>
  );
}

// --- Extracted Entity Inspector (shows "Return to group" button) ---

function ExtractedEntityInspector({
  entityName,
}: {
  entityName: string;
  groupId: string;
}) {
  const retractEntity = useSimStore((s) => s.retractEntity);
  const state = useSimStore((s) => s.state);
  const [entityState, setEntityState] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    fetch(`/api/entity_state?entity=${encodeURIComponent(entityName)}`)
      .then((r) => r.json())
      .then((data: { entity: string; state: Record<string, unknown> }) => {
        setEntityState(data.state);
      });
  }, [entityName, state?.events_processed]);

  return (
    <div className="p-3 space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-white">{entityName}</h3>
        <span className="text-xs text-gray-500">Extracted from group</span>
      </div>

      {entityState && (
        <div className="space-y-1">
          {Object.entries(entityState).map(([key, value]) => (
            <div key={key} className="flex justify-between text-xs">
              <span className="text-gray-400">{key}</span>
              <span className="text-white font-mono">{formatValue(value)}</span>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={() => retractEntity(entityName)}
        className="w-full px-2 py-1 text-xs text-amber-400 hover:text-amber-300 hover:bg-gray-800 rounded border border-gray-700"
      >
        Return to group
      </button>
    </div>
  );
}

// --- Main Inspector Panel ---

export default function InspectorPanel() {
  const state = useSimStore((s) => s.state);
  const selectedEntity = useSimStore((s) => s.selectedEntity);
  const topology = useSimStore((s) => s.topology);
  const dashboardPanels = useSimStore((s) => s.dashboardPanels);
  const addDashboardPanel = useSimStore((s) => s.addDashboardPanel);
  const setActiveView = useSimStore((s) => s.setActiveView);
  const extractedEntities = useSimStore((s) => s.extractedEntities);
  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const [historyData, setHistoryData] = useState<EntityHistoryData | null>(null);
  const [metricModes, setMetricModes] = useState<Record<string, MetricMode>>({});
  const lastFetchedRef = useRef<{ name: string; events: number } | null>(null);
  const lastHistoryFetchRef = useRef<{ name: string; events: number } | null>(null);
  const prevEntityRef = useRef<string | null>(null);

  const entityName = selectedEntity;
  const isGroup = entityName?.startsWith("group:") ?? false;
  const isExtracted = entityName ? extractedEntities.has(entityName) : false;
  const entityData = entityName && !isGroup ? state?.entities[entityName] : null;
  const nodeInfo = topology?.nodes.find((n) => n.id === entityName);
  const isProbe = nodeInfo?.category === "probe";
  const sourceProfile = nodeInfo?.profile;

  // Reset metric display modes when entity changes
  useEffect(() => {
    if (entityName !== prevEntityRef.current) {
      prevEntityRef.current = entityName;
      setMetricModes({});
    }
  }, [entityName]);

  // Fetch time series when a probe is selected or state updates
  useEffect(() => {
    if (!isProbe || !entityName || !state) {
      setTsData(null);
      lastFetchedRef.current = null;
      return;
    }

    // Avoid redundant fetches if nothing changed
    const key = { name: entityName, events: state.events_processed };
    if (
      lastFetchedRef.current &&
      lastFetchedRef.current.name === key.name &&
      lastFetchedRef.current.events === key.events
    ) {
      return;
    }

    lastFetchedRef.current = key;
    fetch(`/api/timeseries?probe=${encodeURIComponent(entityName)}`)
      .then((r) => r.json())
      .then((data: TimeSeriesData) => setTsData(data));
  }, [isProbe, entityName, state?.events_processed, state]);

  // Fetch entity history for sparklines on any selected entity
  useEffect(() => {
    if (!entityName || isGroup || !state) {
      setHistoryData(null);
      lastHistoryFetchRef.current = null;
      return;
    }

    const key = { name: entityName, events: state.events_processed };
    if (
      lastHistoryFetchRef.current &&
      lastHistoryFetchRef.current.name === key.name &&
      lastHistoryFetchRef.current.events === key.events
    ) {
      return;
    }

    lastHistoryFetchRef.current = key;
    fetch(`/api/entity_history?entity=${encodeURIComponent(entityName)}`)
      .then((r) => r.json())
      .then((data: EntityHistoryData) => setHistoryData(data));
  }, [entityName, isGroup, state?.events_processed, state]);

  if (!state) return null;

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Inspector
        </span>
      </div>

      {/* Group inspector */}
      {entityName && isGroup && nodeInfo ? (
        <GroupInspector
          groupId={entityName}
          nodeInfo={{
            type: nodeInfo.type,
            category: nodeInfo.category,
            member_count: nodeInfo.member_count,
          }}
        />
      ) : entityName && isExtracted ? (
        <ExtractedEntityInspector
          entityName={entityName}
          groupId={extractedEntities.get(entityName)!}
        />
      ) : entityName && entityData ? (
        <div className="p-3 space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-white">{entityName}</h3>
            {nodeInfo && (
              <span className="text-xs text-gray-500">{nodeInfo.type}</span>
            )}
          </div>
          <div className="space-y-1">
            {Object.entries(entityData).map(([key, value]) => {
              const history = historyData?.metrics[key];
              const hasSparkline = typeof value === "number" && history && history.values.length >= 2;
              const mode = metricModes[key] ?? "total";
              let sparklineValues: number[] | undefined;
              if (hasSparkline) {
                const raw: Series = { times: history.times, values: history.values };
                let transformed: Series;
                switch (mode) {
                  case "rate":
                    transformed = toRate(raw);
                    break;
                  case "avg":
                    transformed = toBucketed(raw, 1.0, "avg");
                    break;
                  case "p99":
                    transformed = toBucketed(raw, 1.0, "p99");
                    break;
                  default:
                    transformed = raw;
                }
                sparklineValues = transformed.values;
              }
              return (
                <div
                  key={key}
                  draggable={!!hasSparkline}
                  onDragStart={hasSparkline ? (e) => {
                    e.dataTransfer.setData(DRAG_MIME, JSON.stringify({
                      kind: "entity_metric",
                      entityName,
                      metricKey: key,
                      displayMode: mode,
                      label: `${entityName}.${key}`,
                    }));
                    e.dataTransfer.effectAllowed = "copy";
                  } : undefined}
                  title={hasSparkline ? "Drag to graph to pin" : undefined}
                  className={hasSparkline ? "cursor-grab active:cursor-grabbing" : undefined}
                >
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">
                      {key}
                      {hasSparkline && (
                        <span
                          className="ml-1 text-gray-500 hover:text-gray-300 cursor-pointer select-none"
                          onClick={() =>
                            setMetricModes((prev) => ({
                              ...prev,
                              [key]: MODE_CYCLE[(MODE_CYCLE.indexOf(mode) + 1) % MODE_CYCLE.length],
                            }))
                          }
                        >
                          {MODE_LABELS[mode]}
                        </span>
                      )}
                    </span>
                    <span className="text-white font-mono">
                      {formatValue(value)}
                    </span>
                  </div>
                  {sparklineValues && sparklineValues.length >= 2 && (
                    <MiniSparkline values={sparklineValues} />
                  )}
                </div>
              );
            })}
          </div>

          {/* Time series chart for probes */}
          {isProbe && tsData && tsData.times.length > 0 && (
            <div
              className="border-t border-gray-800 pt-3 cursor-grab active:cursor-grabbing"
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(DRAG_MIME, JSON.stringify({
                  kind: "probe",
                  probeName: entityName,
                  label: `${tsData.target}.${tsData.metric}`,
                }));
                e.dataTransfer.effectAllowed = "copy";
              }}
              title="Drag to graph to pin"
            >
              <TimeSeriesChart
                times={tsData.times}
                values={tsData.values}
                label={`${tsData.target}.${tsData.metric}`}
                color="#3b82f6"
              />
              {!dashboardPanels.some((p) => p.probeName === entityName) && (
                <button
                  onClick={() => {
                    addDashboardPanel({
                      id: crypto.randomUUID(),
                      probeName: entityName!,
                      label: `${tsData.target}.${tsData.metric}`,
                      x: (dashboardPanels.length % 3) * 4,
                      y: Math.floor(dashboardPanels.length / 3) * 4,
                      w: 4,
                      h: 4,
                    });
                    setActiveView("dashboard");
                  }}
                  className="mt-2 w-full px-2 py-1 text-xs text-blue-400 hover:text-blue-300 hover:bg-gray-800 rounded border border-gray-700"
                >
                  + Dashboard
                </button>
              )}
            </div>
          )}

          {/* Load profile chart for sources */}
          {sourceProfile && sourceProfile.times.length > 0 && (
            <div className="border-t border-gray-800 pt-3">
              <div className="text-xs font-semibold text-gray-400 mb-1">Load Profile</div>
              <TimeSeriesChart
                times={sourceProfile.times}
                values={sourceProfile.values}
                label="Rate (req/s)"
                color="#22c55e"
                yLabel="req/s"
                xLabel="Time (s)"
              />
            </div>
          )}
        </div>
      ) : (
        <div className="p-3 text-xs text-gray-500">
          Click a node to inspect its state
        </div>
      )}

      {/* Upcoming events */}
      {state.upcoming.length > 0 && (
        <div className="border-t border-gray-800 mt-auto">
          <div className="px-3 py-2">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
              Upcoming
            </span>
          </div>
          <div className="text-xs font-mono px-3 pb-2 space-y-1 h-[7.5rem] overflow-y-auto">
            {state.upcoming.map((e, i) => (
              <div key={i} className="flex gap-2 text-gray-400">
                <span className="w-14 text-right text-gray-500">
                  {e.time_s.toFixed(4)}
                </span>
                <span className="text-blue-400 truncate">{e.event_type}</span>
                <span className="text-gray-600">&rarr;</span>
                <span className="text-emerald-400 truncate">{e.target}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatValue(v: unknown): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  }
  if (typeof v === "object" && v !== null) {
    return JSON.stringify(v);
  }
  return String(v);
}
