import { useEffect, useState, useRef } from "react";
import { useSimStore } from "../hooks/useSimState";
import TimeSeriesChart from "./TimeSeriesChart";

interface TimeSeriesData {
  name: string;
  metric: string;
  target: string;
  times: number[];
  values: number[];
}

export default function InspectorPanel() {
  const state = useSimStore((s) => s.state);
  const selectedEntity = useSimStore((s) => s.selectedEntity);
  const topology = useSimStore((s) => s.topology);
  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const lastFetchedRef = useRef<{ name: string; events: number } | null>(null);

  const entityName = selectedEntity;
  const entityData = entityName ? state?.entities[entityName] : null;
  const nodeInfo = topology?.nodes.find((n) => n.id === entityName);
  const isProbe = nodeInfo?.category === "probe";

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
            {Object.entries(entityData).map(([key, value]) => (
              <div key={key} className="flex justify-between text-xs">
                <span className="text-gray-400">{key}</span>
                <span className="text-white font-mono">
                  {formatValue(value)}
                </span>
              </div>
            ))}
          </div>

          {/* Time series chart for probes */}
          {isProbe && tsData && tsData.times.length > 0 && (
            <div className="border-t border-gray-800 pt-3">
              <TimeSeriesChart
                times={tsData.times}
                values={tsData.values}
                label={`${tsData.target}.${tsData.metric}`}
                color="#3b82f6"
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
