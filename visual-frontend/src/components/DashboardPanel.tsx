import { useEffect, useState, useRef } from "react";
import { useSimStore } from "../hooks/useSimState";
import TimeSeriesChart from "./TimeSeriesChart";
import type { ChartConfig } from "../types";
import { exportCanvasPng, exportCsv } from "../utils/export";

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

export default function DashboardPanel({ probeName, chartConfig, label, onClose }: Props) {
  const eventsProcessed = useSimStore((s) => s.state?.events_processed);
  const timeRange = useSimStore((s) => s.dashboardTimeRange);
  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const lastFetchedRef = useRef<string | null>(null);
  const chartRef = useRef<{ getCanvas: () => HTMLCanvasElement | null }>(null);

  const sanitizedLabel = label.replace(/[^a-zA-Z0-9]/g, "_");

  const handlePngExport = () => {
    const canvas = chartRef.current?.getCanvas();
    if (canvas) exportCanvasPng(canvas, sanitizedLabel + ".png");
  };

  const handleCsvExport = () => {
    if (tsData) exportCsv(tsData.times, tsData.values, sanitizedLabel + ".csv");
  };

  // Reset cached data when events_processed drops (i.e. simulation was reset)
  const prevEventsRef = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (prevEventsRef.current !== undefined && eventsProcessed !== undefined && eventsProcessed < prevEventsRef.current) {
      lastFetchedRef.current = null;
      setTsData(null);
    }
    prevEventsRef.current = eventsProcessed;
  }, [eventsProcessed]);

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

  const chartColor = chartConfig?.color ?? "#3b82f6";

  return (
    <div className="h-full flex flex-col bg-gray-900 border border-gray-700 rounded-lg shadow-lg overflow-hidden">
      {/* Title bar — drag handle */}
      <div className="drag-handle flex items-center justify-between px-3 py-1.5 border-b border-gray-700 cursor-grab active:cursor-grabbing select-none shrink-0">
        <span className="text-xs font-semibold text-gray-300 truncate">{label}</span>
        <div className="flex items-center gap-1">
          <button onClick={handlePngExport} disabled={!tsData?.times.length} className="text-gray-500 hover:text-gray-300 disabled:opacity-30 text-[10px] px-1" title="Export PNG">PNG</button>
          <button onClick={handleCsvExport} disabled={!tsData?.times.length} className="text-gray-500 hover:text-gray-300 disabled:opacity-30 text-[10px] px-1" title="Export CSV">CSV</button>
          <button onClick={onClose} className="text-gray-500 hover:text-red-400 text-xs ml-1 leading-none">&times;</button>
        </div>
      </div>
      {/* Chart body — fills remaining space */}
      <div className="flex-1 min-h-0 p-2">
        {tsData && tsData.times.length > 0 ? (
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
