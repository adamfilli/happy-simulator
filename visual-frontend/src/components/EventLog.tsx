import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { useSimStore } from "../hooks/useSimState";

export default function EventLog() {
  const eventLog = useSimStore((s) => s.eventLog);
  const showInternal = useSimStore((s) => s.showInternal);
  const toggleInternal = useSimStore((s) => s.toggleInternal);
  const endRef = useRef<HTMLDivElement>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  // Filter state
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [entityFilter, setEntityFilter] = useState("");
  const [searchText, setSearchText] = useState("");
  const [timeMin, setTimeMin] = useState("");
  const [timeMax, setTimeMax] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [showFilters, setShowFilters] = useState(false);
  const [typeDropdownOpen, setTypeDropdownOpen] = useState(false);

  // Unique event types for multi-select
  const eventTypes = useMemo(() => {
    const types = new Set<string>();
    for (const e of eventLog) types.add(e.event_type);
    return Array.from(types).sort();
  }, [eventLog]);

  const hasActiveFilters =
    selectedTypes.size > 0 ||
    entityFilter !== "" ||
    searchText !== "" ||
    timeMin !== "" ||
    timeMax !== "";

  const clearFilters = useCallback(() => {
    setSelectedTypes(new Set());
    setEntityFilter("");
    setSearchText("");
    setTimeMin("");
    setTimeMax("");
  }, []);

  const filtered = useMemo(() => {
    let result = showInternal
      ? eventLog
      : eventLog.filter((e) => !e.is_internal);

    if (selectedTypes.size > 0) {
      result = result.filter((e) => selectedTypes.has(e.event_type));
    }

    if (entityFilter) {
      const lower = entityFilter.toLowerCase();
      result = result.filter(
        (e) =>
          e.target_name.toLowerCase().includes(lower) ||
          (e.source_name && e.source_name.toLowerCase().includes(lower))
      );
    }

    if (searchText) {
      const lower = searchText.toLowerCase();
      result = result.filter((e) => {
        if (e.event_type.toLowerCase().includes(lower)) return true;
        if (e.target_name.toLowerCase().includes(lower)) return true;
        if (e.source_name && e.source_name.toLowerCase().includes(lower))
          return true;
        if (String(e.event_id).includes(lower)) return true;
        if (e.context) {
          const json = JSON.stringify(e.context).toLowerCase();
          if (json.includes(lower)) return true;
        }
        return false;
      });
    }

    const tMin = timeMin !== "" ? parseFloat(timeMin) : null;
    const tMax = timeMax !== "" ? parseFloat(timeMax) : null;
    if (tMin !== null && !isNaN(tMin)) {
      result = result.filter((e) => e.time_s >= tMin);
    }
    if (tMax !== null && !isNaN(tMax)) {
      result = result.filter((e) => e.time_s <= tMax);
    }

    return result;
  }, [
    eventLog,
    showInternal,
    selectedTypes,
    entityFilter,
    searchText,
    timeMin,
    timeMax,
  ]);

  useEffect(() => {
    if (autoScroll) {
      endRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [filtered.length, autoScroll]);

  const toggleExpand = (eventId: number) => {
    setExpandedId((prev) => (prev === eventId ? null : eventId));
  };

  const toggleType = (type: string) => {
    setSelectedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
            Event Log ({filtered.length})
          </span>
          <button
            onClick={() => setShowFilters((v) => !v)}
            className={`text-[10px] px-1.5 py-0.5 rounded ${
              showFilters || hasActiveFilters
                ? "bg-blue-900/50 text-blue-400"
                : "bg-gray-800 text-gray-500 hover:text-gray-400"
            }`}
            title="Toggle filters"
          >
            ⛶ Filter{hasActiveFilters ? " ●" : ""}
          </button>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-xs text-gray-500 cursor-pointer" title="Auto-scroll to latest events">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={() => setAutoScroll((v) => !v)}
              className="rounded"
            />
            <span className="text-[10px]">Auto-scroll</span>
          </label>
          <label className="flex items-center gap-1 text-xs text-gray-500 cursor-pointer">
            <input
              type="checkbox"
              checked={showInternal}
              onChange={toggleInternal}
              className="rounded"
            />
            <span className="text-[10px]">Internal</span>
          </label>
        </div>
      </div>

      {/* Filter toolbar */}
      {showFilters && (
        <div className="px-3 py-2 border-b border-gray-800 bg-gray-900/50 space-y-2">
          {/* Row 1: Event type multi-select + Entity filter */}
          <div className="flex gap-2 items-start">
            {/* Event type multi-select dropdown */}
            <div className="relative flex-1 min-w-0">
              <button
                onClick={() => setTypeDropdownOpen((v) => !v)}
                className="w-full flex items-center justify-between px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-300 hover:border-gray-600"
              >
                <span className="truncate">
                  {selectedTypes.size === 0
                    ? "All event types"
                    : `${selectedTypes.size} type${selectedTypes.size > 1 ? "s" : ""} selected`}
                </span>
                <span className="text-gray-500 ml-1 shrink-0">▾</span>
              </button>
              {typeDropdownOpen && (
                <div className="absolute z-50 mt-1 w-full max-h-48 overflow-y-auto bg-gray-800 border border-gray-700 rounded shadow-lg">
                  {eventTypes.length === 0 ? (
                    <div className="px-2 py-1.5 text-xs text-gray-500">
                      No events yet
                    </div>
                  ) : (
                    eventTypes.map((type) => (
                      <label
                        key={type}
                        className="flex items-center gap-2 px-2 py-1 text-xs text-gray-300 hover:bg-gray-700/50 cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          checked={selectedTypes.has(type)}
                          onChange={() => toggleType(type)}
                          className="rounded"
                        />
                        <span className="truncate">{type}</span>
                      </label>
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Entity text filter */}
            <input
              type="text"
              placeholder="Entity filter…"
              value={entityFilter}
              onChange={(e) => setEntityFilter(e.target.value)}
              className="flex-1 min-w-0 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-300 placeholder-gray-600"
            />
          </div>

          {/* Row 2: Full-text search + Time range */}
          <div className="flex gap-2 items-center">
            <input
              type="text"
              placeholder="Search…"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              className="flex-1 min-w-0 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-300 placeholder-gray-600"
            />
            <input
              type="number"
              placeholder="t≥"
              value={timeMin}
              onChange={(e) => setTimeMin(e.target.value)}
              step="any"
              className="w-16 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-300 placeholder-gray-600"
              title="Minimum time (seconds)"
            />
            <input
              type="number"
              placeholder="t≤"
              value={timeMax}
              onChange={(e) => setTimeMax(e.target.value)}
              step="any"
              className="w-16 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-300 placeholder-gray-600"
              title="Maximum time (seconds)"
            />
            {hasActiveFilters && (
              <button
                onClick={clearFilters}
                className="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-300 rounded text-xs shrink-0"
                title="Clear all filters"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      {/* Event list */}
      <div
        className="flex-1 overflow-y-auto text-xs font-mono"
        onClick={(e) => {
          // Close type dropdown when clicking outside it
          if (typeDropdownOpen) {
            const target = e.target as HTMLElement;
            if (!target.closest(".relative")) {
              setTypeDropdownOpen(false);
            }
          }
        }}
      >
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
                <ContextPanel context={e.context} event={e} />
              )}
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function ContextPanel({
  context,
  event,
}: {
  context: Record<string, unknown>;
  event: { time_s: number; event_type: string; target_name: string; source_name: string | null; event_id: number };
}) {
  const [copied, setCopied] = useState(false);

  // Separate trace spans from other context fields
  const trace = context.trace as { spans?: unknown[] } | undefined;
  const spans = trace?.spans;
  const otherKeys = Object.keys(context).filter((k) => {
    if (k === "trace") return false;
    const v = context[k];
    if (Array.isArray(v) && v.length === 0) return false;
    if (typeof v === "object" && v !== null && !Array.isArray(v) && Object.keys(v).length === 0) return false;
    return true;
  });

  const copyToClipboard = () => {
    const lines: string[] = [
      `Event #${event.event_id}`,
      `Time: ${event.time_s.toFixed(6)}`,
      `Type: ${event.event_type}`,
      `Target: ${event.target_name}`,
    ];
    if (event.source_name) lines.push(`Source: ${event.source_name}`);
    if (otherKeys.length > 0) {
      lines.push("Context:");
      for (const key of otherKeys) {
        lines.push(`  ${key}: ${formatContextValue(context[key])}`);
      }
    }
    if (spans && spans.length > 0) {
      lines.push("Trace Spans:");
      for (const span of spans as Record<string, unknown>[]) {
        const time =
          typeof span.time === "number" ? span.time.toFixed(6) : String(span.time ?? "");
        let line = `  ${String(span.action ?? "")} @ ${time}`;
        if ("data" in span && span.data != null) {
          line += ` — ${formatContextValue(span.data)}`;
        }
        lines.push(line);
      }
    }
    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="px-4 py-2 bg-gray-900/80 border-b border-gray-800 space-y-2">
      {/* Copy button */}
      <div className="flex justify-end">
        <button
          onClick={copyToClipboard}
          className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-500 hover:text-gray-300"
          title="Copy to clipboard"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>

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
