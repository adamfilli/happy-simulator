import { useEffect, useState, useRef } from "react";
import { useSimStore } from "../hooks/useSimState";
import TimeSeriesChart from "./TimeSeriesChart";
import MiniSparkline from "./MiniSparkline";
import { toRate, toBucketed, type Series } from "./sparklineTransforms";

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

export default function InspectorPanel() {
  const state = useSimStore((s) => s.state);
  const selectedEntity = useSimStore((s) => s.selectedEntity);
  const topology = useSimStore((s) => s.topology);
  const dashboardPanels = useSimStore((s) => s.dashboardPanels);
  const addDashboardPanel = useSimStore((s) => s.addDashboardPanel);
  const setActiveView = useSimStore((s) => s.setActiveView);
  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const [historyData, setHistoryData] = useState<EntityHistoryData | null>(null);
  const [metricModes, setMetricModes] = useState<Record<string, MetricMode>>({});
  const lastFetchedRef = useRef<{ name: string; events: number } | null>(null);
  const lastHistoryFetchRef = useRef<{ name: string; events: number } | null>(null);
  const prevEntityRef = useRef<string | null>(null);

  const entityName = selectedEntity;
  const entityData = entityName ? state?.entities[entityName] : null;
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
    if (!entityName || !state) {
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
  }, [entityName, state?.events_processed, state]);

  if (!state) return null;

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Inspector
        </span>
      </div>

      {entityName && entityData ? (
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
          <div className="text-xs font-mono px-3 pb-2 space-y-1">
            {state.upcoming.slice(0, 8).map((e, i) => (
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
