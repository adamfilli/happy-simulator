import { useRef, useEffect, useState } from "react";
import { useSimStore } from "../hooks/useSimState";

export default function EventLog() {
  const eventLog = useSimStore((s) => s.eventLog);
  const showInternal = useSimStore((s) => s.showInternal);
  const toggleInternal = useSimStore((s) => s.toggleInternal);
  const endRef = useRef<HTMLDivElement>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const filtered = showInternal
    ? eventLog
    : eventLog.filter((e) => !e.is_internal);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [filtered.length]);

  const toggleExpand = (eventId: number) => {
    setExpandedId((prev) => (prev === eventId ? null : eventId));
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Event Log ({filtered.length})
        </span>
        <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
          <input
            type="checkbox"
            checked={showInternal}
            onChange={toggleInternal}
            className="rounded"
          />
          Internal
        </label>
      </div>
      <div className="flex-1 overflow-y-auto text-xs font-mono">
        {filtered.slice(-200).map((e, i) => {
          const isExpanded = expandedId === e.event_id;
          return (
            <div key={`${e.event_id}-${i}`}>
              <div
                onClick={() => toggleExpand(e.event_id)}
                className={`px-3 py-1 border-b border-gray-900 flex gap-2 cursor-pointer hover:bg-gray-800/50 ${
                  e.is_internal ? "text-gray-600" : "text-gray-300"
                } ${isExpanded ? "bg-gray-800/30" : ""}`}
              >
                <span className="text-gray-600 shrink-0 w-4">
                  {e.context ? (isExpanded ? "▼" : "▶") : " "}
                </span>
                <span className="text-gray-500 w-16 shrink-0 text-right">
                  {e.time_s.toFixed(4)}
                </span>
                <span className="text-blue-400 truncate w-24 shrink-0">{e.event_type}</span>
                {e.source_name && (
                  <>
                    <span className="text-gray-600">{e.source_name}</span>
                    <span className="text-gray-700">&rarr;</span>
                  </>
                )}
                <span className="text-emerald-400 truncate">{e.target_name}</span>
                <span className="text-gray-600 ml-auto shrink-0">#{e.event_id}</span>
              </div>
              {isExpanded && e.context && (
                <ContextPanel context={e.context} />
              )}
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function ContextPanel({ context }: { context: Record<string, unknown> }) {
  // Separate trace spans from other context fields
  const trace = context.trace as { spans?: unknown[] } | undefined;
  const spans = trace?.spans;
  const otherKeys = Object.keys(context).filter((k) => {
    if (k === "trace") return false;
    const v = context[k];
    // Hide empty arrays and objects
    if (Array.isArray(v) && v.length === 0) return false;
    if (typeof v === "object" && v !== null && !Array.isArray(v) && Object.keys(v).length === 0) return false;
    return true;
  });

  return (
    <div className="px-4 py-2 bg-gray-900/80 border-b border-gray-800 space-y-2">
      {/* Key-value pairs */}
      {otherKeys.length > 0 && (
        <div className="space-y-0.5">
          {otherKeys.map((key) => (
            <div key={key} className="flex gap-2 text-xs">
              <span className="text-gray-500 shrink-0">{key}:</span>
              <span className="text-gray-300 break-all">
                {formatContextValue(context[key])}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Trace spans */}
      {spans && spans.length > 0 && (
        <div>
          <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">
            Trace Spans
          </div>
          <div className="space-y-0.5">
            {(spans as Record<string, unknown>[]).map((span, i) => (
              <div
                key={i}
                className="flex gap-2 text-xs pl-2 border-l-2 border-gray-700"
              >
                <span className="text-yellow-500 shrink-0 w-32 truncate">
                  {String(span.action ?? "")}
                </span>
                <span className="text-gray-500 w-16 text-right shrink-0">
                  {typeof span.time === "number"
                    ? span.time.toFixed(4)
                    : String(span.time ?? "")}
                </span>
                {"data" in span && span.data != null && (
                  <span className="text-gray-400 truncate">
                    {formatContextValue(span.data)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatContextValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(6);
  }
  if (typeof v === "string") return v;
  if (typeof v === "boolean") return String(v);
  if (Array.isArray(v)) {
    if (v.length === 0) return "[]";
    return JSON.stringify(v);
  }
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    if (entries.length === 0) return "{}";
    return entries
      .map(([k, val]) => `${k}=${formatContextValue(val)}`)
      .join(", ");
  }
  return String(v);
}
