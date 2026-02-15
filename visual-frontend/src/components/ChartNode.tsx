import { memo, useEffect, useRef, useState } from "react";
import { type NodeProps, type Node } from "@xyflow/react";
import { useSimStore } from "../hooks/useSimState";
import TimeSeriesChart from "./TimeSeriesChart";
import { toRate, toBucketed, type Series } from "./sparklineTransforms";

type DisplayMode = "total" | "rate" | "avg" | "p99";
const MODE_CYCLE: DisplayMode[] = ["total", "rate", "avg", "p99"];
const MODE_LABELS: Record<DisplayMode, string> = {
  total: "total",
  rate: "/s",
  avg: "avg",
  p99: "p99",
};

type ChartNodeData = {
  chartId: string;
  kind: "entity_metric" | "probe";
  entityName?: string;
  metricKey?: string;
  displayMode?: DisplayMode;
  probeName?: string;
  label: string;
};

type ChartNodeType = Node<ChartNodeData, "chart">;

interface TimeSeriesData {
  times: number[];
  values: number[];
}

function ChartNode({ data }: NodeProps<ChartNodeType>) {
  const { chartId, kind, entityName, metricKey, probeName, label } = data;
  const displayMode = data.displayMode ?? "total";

  const eventsProcessed = useSimStore((s) => s.state?.events_processed);
  const removePinnedChart = useSimStore((s) => s.removePinnedChart);
  const updatePinnedChartMode = useSimStore((s) => s.updatePinnedChartMode);

  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const lastFetchedRef = useRef<string | null>(null);

  // Fetch data keyed on events_processed
  useEffect(() => {
    if (eventsProcessed === undefined) return;
    const key = `${eventsProcessed}`;
    if (lastFetchedRef.current === key) return;
    lastFetchedRef.current = key;

    if (kind === "entity_metric" && entityName) {
      fetch(`/api/entity_history?entity=${encodeURIComponent(entityName)}`)
        .then((r) => r.json())
        .then((data: { metrics: Record<string, { times: number[]; values: number[] }> }) => {
          const metric = metricKey ? data.metrics[metricKey] : null;
          if (metric) {
            setTsData({ times: metric.times, values: metric.values });
          }
        })
        .catch(() => {});
    } else if (kind === "probe" && probeName) {
      fetch(`/api/timeseries?probe=${encodeURIComponent(probeName)}`)
        .then((r) => r.json())
        .then((data: { times: number[]; values: number[] }) => {
          setTsData({ times: data.times, values: data.values });
        })
        .catch(() => {});
    }
  }, [eventsProcessed, kind, entityName, metricKey, probeName]);

  // Apply transform for entity metrics
  let displayTimes = tsData?.times ?? [];
  let displayValues = tsData?.values ?? [];
  if (tsData && kind === "entity_metric" && displayMode !== "total") {
    const raw: Series = { times: tsData.times, values: tsData.values };
    let transformed: Series;
    switch (displayMode) {
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
    displayTimes = transformed.times;
    displayValues = transformed.values;
  }

  const hasData = displayTimes.length > 0;
  const chartColor = kind === "entity_metric" ? "#22c55e" : "#3b82f6";

  const cycleMode = () => {
    const next = MODE_CYCLE[(MODE_CYCLE.indexOf(displayMode) + 1) % MODE_CYCLE.length];
    updatePinnedChartMode(chartId, next);
  };

  return (
    <div
      className="bg-gray-900 border border-gray-700 rounded-lg shadow-lg overflow-hidden"
      style={{ width: 280, height: 170 }}
    >
      {/* Title bar - 28px */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-gray-700 cursor-grab active:cursor-grabbing select-none"
        style={{ height: 28 }}
      >
        <span className="text-[10px] font-semibold text-gray-300 truncate flex-1 mr-1">
          {label}
        </span>
        <div className="flex items-center gap-1 shrink-0">
          {kind === "entity_metric" && (
            <button
              onClick={(e) => { e.stopPropagation(); cycleMode(); }}
              className="nodrag text-[9px] text-gray-500 hover:text-gray-300 px-1 rounded hover:bg-gray-800"
            >
              {MODE_LABELS[displayMode]}
            </button>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); removePinnedChart(chartId); }}
            className="nodrag text-gray-500 hover:text-red-400 text-xs leading-none px-0.5"
          >
            &times;
          </button>
        </div>
      </div>
      {/* Chart area */}
      <div className="nodrag nowheel p-1" style={{ height: 140 }}>
        {hasData ? (
          <TimeSeriesChart
            times={displayTimes}
            values={displayValues}
            label={label}
            color={chartColor}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-[10px] text-gray-600">
            No data yet
          </div>
        )}
      </div>
    </div>
  );
}

export default memo(ChartNode);
