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

interface Props {
  id: string;
  probeName: string;
  label: string;
  onClose: () => void;
}

export default function DashboardPanel({ probeName, label, onClose }: Props) {
  const eventsProcessed = useSimStore((s) => s.state?.events_processed);
  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const lastFetchedRef = useRef<number | null>(null);

  useEffect(() => {
    if (eventsProcessed === undefined) return;
    if (lastFetchedRef.current === eventsProcessed) return;
    lastFetchedRef.current = eventsProcessed;

    fetch(`/api/timeseries?probe=${encodeURIComponent(probeName)}`)
      .then((r) => r.json())
      .then((data: TimeSeriesData) => setTsData(data))
      .catch(() => {});
  }, [probeName, eventsProcessed]);

  return (
    <div className="h-full flex flex-col bg-gray-900 border border-gray-700 rounded-lg shadow-lg overflow-hidden">
      {/* Title bar — drag handle */}
      <div className="drag-handle flex items-center justify-between px-3 py-1.5 border-b border-gray-700 cursor-grab active:cursor-grabbing select-none shrink-0">
        <span className="text-xs font-semibold text-gray-300 truncate">{label}</span>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-red-400 text-xs ml-2 leading-none"
        >
          &times;
        </button>
      </div>
      {/* Chart body — fills remaining space */}
      <div className="flex-1 min-h-0 p-2">
        {tsData && tsData.times.length > 0 ? (
          <TimeSeriesChart
            times={tsData.times}
            values={tsData.values}
            label={label}
            color="#3b82f6"
          />
        ) : (
          <div className="text-xs text-gray-500 text-center py-8">No data yet</div>
        )}
      </div>
    </div>
  );
}
