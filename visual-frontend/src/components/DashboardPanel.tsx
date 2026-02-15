import { useEffect, useState, useRef, useCallback } from "react";
import { useSimStore } from "../hooks/useSimState";
import TimeSeriesChart from "./TimeSeriesChart";
import type { TimeSeriesChartHandle } from "./TimeSeriesChart";
import type { ChartConfig } from "../types";

interface TimeSeriesData {
  times: number[];
  values: number[];
}

interface Props {
  id: string;
  label: string;
  onClose: () => void;
  probeName?: string;
  chartConfig?: ChartConfig;
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]/g, "_");
}

export default function DashboardPanel({ probeName, chartConfig, label, onClose }: Props) {
  const eventsProcessed = useSimStore((s) => s.state?.events_processed);
  const timeRange = useSimStore((s) => s.dashboardTimeRange);
  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const lastFetchedRef = useRef<string | null>(null);
  const chartRef = useRef<TimeSeriesChartHandle>(null);

  useEffect(() => {
    if (eventsProcessed === undefined) return;
    const key = `${eventsProcessed}:${timeRange.start}:${timeRange.end}`;
    if (lastFetchedRef.current === key) return;
    lastFetchedRef.current = key;

    const params = new URLSearchParams();
    if (chartConfig) {
      params.set("chart_id", chartConfig.chart_id);
    } else {
      params.set("probe", probeName!);
    }
    if (timeRange.start != null) params.set("start_s", String(timeRange.start));
    if (timeRange.end != null) params.set("end_s", String(timeRange.end));

    const endpoint = chartConfig ? "/api/chart_data" : "/api/timeseries";
    fetch(`${endpoint}?${params}`)
      .then((r) => r.json())
      .then((data) => setTsData({ times: data.times, values: data.values }))
      .catch(() => {});
  }, [probeName, chartConfig, eventsProcessed, timeRange]);

  const exportPng = useCallback(() => {
    const canvas = chartRef.current?.getCanvas();
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${sanitizeFilename(label)}.png`;
      a.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }, [label]);

  const exportCsv = useCallback(() => {
    if (!tsData || tsData.times.length === 0) return;
    const header = `Time (s),${label}\n`;
    const rows = tsData.times.map((t, i) => `${t},${tsData.values[i]}`).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sanitizeFilename(label)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [tsData, label]);

  const chartColor = chartConfig?.color ?? "#3b82f6";
  const hasData = tsData && tsData.times.length > 0;

  return (
    <div className="h-full flex flex-col bg-gray-900 border border-gray-700 rounded-lg shadow-lg overflow-hidden">
      {/* Title bar — drag handle */}
      <div className="drag-handle flex items-center justify-between px-3 py-1.5 border-b border-gray-700 cursor-grab active:cursor-grabbing select-none shrink-0">
        <span className="text-xs font-semibold text-gray-300 truncate">{label}</span>
        <div className="flex items-center gap-1 ml-2 shrink-0">
          {hasData && (
            <>
              <button
                onClick={exportPng}
                title="Export PNG"
                className="text-gray-500 hover:text-gray-300 text-[10px] leading-none px-1"
              >
                PNG
              </button>
              <button
                onClick={exportCsv}
                title="Export CSV"
                className="text-gray-500 hover:text-gray-300 text-[10px] leading-none px-1"
              >
                CSV
              </button>
            </>
          )}
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-red-400 text-xs leading-none"
          >
            &times;
          </button>
        </div>
      </div>
      {/* Chart body — fills remaining space */}
      <div className="flex-1 min-h-0 p-2">
        {hasData ? (
          <TimeSeriesChart
            ref={chartRef}
            times={tsData.times}
            values={tsData.values}
            label={label}
            color={chartColor}
            yLabel={chartConfig?.y_label}
            xLabel={chartConfig?.x_label}
            yMin={chartConfig?.y_min}
            yMax={chartConfig?.y_max}
          />
        ) : (
          <div className="text-xs text-gray-500 text-center py-8">No data yet</div>
        )}
      </div>
    </div>
  );
}
