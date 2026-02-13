import { useEffect, useState, useRef, useCallback } from "react";
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
  x: number;
  y: number;
}

export default function DashboardPanel({ id, probeName, label, x, y }: Props) {
  const eventsProcessed = useSimStore((s) => s.state?.events_processed);
  const moveDashboardPanel = useSimStore((s) => s.moveDashboardPanel);
  const removeDashboardPanel = useSimStore((s) => s.removeDashboardPanel);
  const [tsData, setTsData] = useState<TimeSeriesData | null>(null);
  const lastFetchedRef = useRef<number | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);

  // Fetch time series data when events_processed changes
  useEffect(() => {
    if (eventsProcessed === undefined) return;
    if (lastFetchedRef.current === eventsProcessed) return;
    lastFetchedRef.current = eventsProcessed;

    fetch(`/api/timeseries?probe=${encodeURIComponent(probeName)}`)
      .then((r) => r.json())
      .then((data: TimeSeriesData) => setTsData(data))
      .catch(() => {});
  }, [probeName, eventsProcessed]);

  const onDragStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragRef.current = { startX: e.clientX, startY: e.clientY, origX: x, origY: y };

      const onMove = (ev: MouseEvent) => {
        if (!dragRef.current) return;
        const dx = ev.clientX - dragRef.current.startX;
        const dy = ev.clientY - dragRef.current.startY;
        moveDashboardPanel(id, dragRef.current.origX + dx, dragRef.current.origY + dy);
      };
      const onUp = () => {
        dragRef.current = null;
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };

      document.body.style.cursor = "grabbing";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [id, x, y, moveDashboardPanel]
  );

  return (
    <div
      className="absolute bg-gray-900 border border-gray-700 rounded-lg shadow-lg"
      style={{ left: x, top: y, width: 400 }}
    >
      {/* Title bar */}
      <div
        onMouseDown={onDragStart}
        className="flex items-center justify-between px-3 py-1.5 border-b border-gray-700 cursor-grab active:cursor-grabbing select-none"
      >
        <span className="text-xs font-semibold text-gray-300 truncate">{label}</span>
        <button
          onClick={() => removeDashboardPanel(id)}
          className="text-gray-500 hover:text-red-400 text-xs ml-2 leading-none"
        >
          &times;
        </button>
      </div>
      {/* Chart body */}
      <div className="p-2">
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
