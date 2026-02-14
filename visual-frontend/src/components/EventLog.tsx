import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { useSimStore } from "../hooks/useSimState";

interface Filters {
  selectedTypes: Set<string>;
  entitySearch: string;
  textSearch: string;
  timeMin: string;
  timeMax: string;
  autoScroll: boolean;
}

const defaultFilters: Filters = {
  selectedTypes: new Set<string>(),
  entitySearch: "",
  textSearch: "",
  timeMin: "",
  timeMax: "",
  autoScroll: true,
};

export default function EventLog() {
  const eventLog = useSimStore((s) => s.eventLog);
  const showInternal = useSimStore((s) => s.showInternal);
  const toggleInternal = useSimStore((s) => s.toggleInternal);
  const endRef = useRef<HTMLDivElement>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [filters, setFilters] = useState<Filters>({ ...defaultFilters });
  const [typeDropdownOpen, setTypeDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setTypeDropdownOpen(false);
      }
    }
    if (typeDropdownOpen) {
      document.addEventListener("mousedown", handleClick);
      return () => document.removeEventListener("mousedown", handleClick);
    }
  }, [typeDropdownOpen]);

  // Derive unique event types from full log
  const eventTypes = useMemo(() => {
    const types = new Set<string>();
    for (const e of eventLog) {
      types.add(e.event_type);
    }
    return Array.from(types).sort();
  }, [eventLog]);

  // Chain all filters
  const filtered = useMemo(() => {
    let result = showInternal
      ? eventLog
      : eventLog.filter((e) => !e.is_internal);

    // Event type filter
    if (filters.selectedTypes.size > 0) {
      result = result.filter((e) => filters.selectedTypes.has(e.event_type));
    }

    // Entity filter (target_name or source_name substring match, case-insensitive)
    if (filters.entitySearch.trim()) {
      const q = filters.entitySearch.trim().toLowerCase();
      result = result.filter(
        (e) =>
          e.target_name.toLowerCase().includes(q) ||
          (e.source_name?.toLowerCase().includes(q) ?? false)
      );
    }

    // Text search across context, event_type, target_name
    if (filters.textSearch.trim()) {
      const q = filters.textSearch.trim().toLowerCase();
      result = result.filter((e) => {
        if (e.event_type.toLowerCase().includes(q)) return true;
        if (e.target_name.toLowerCase().includes(q)) return true;
        if (e.source_name?.toLowerCase().includes(q)) return true;
        if (e.context) {
          try {
            if (JSON.stringify(e.context).toLowerCase().includes(q)) return true;
          } catch {
            // ignore serialization errors
          }
        }
        return false;
      });
    }

    // Time range filter
    const tMin = filters.timeMin !== "" ? parseFloat(filters.timeMin) : NaN;
    const tMax = filters.timeMax !== "" ? parseFloat(filters.timeMax) : NaN;
    if (!isNaN(tMin)) {
      result = result.filter((e) => e.time_s >= tMin);
    }
    if (!isNaN(tMax)) {
      result = result.filter((e) => e.time_s <= tMax);
    }

    return result;
  }, [eventLog, showInternal, filters]);

  // Auto-scroll
  useEffect(() => {
    if (filters.autoScroll) {
      endRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [filtered.length, filters.autoScroll]);

  const toggleExpand = (eventId: number) => {
    setExpandedId((prev) => (prev === eventId ? null : eventId));
  };

  const toggleType = useCallback((type: string) => {
    setFilters((prev) => {
      const next = new Set(prev.selectedTypes);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return { ...prev, selectedTypes: next };
    });
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({ ...defaultFilters, selectedTypes: new Set<string>() });
    setTypeDropdownOpen(false);
  }, []);

  const hasActiveFilters =
    filters.selectedTypes.size > 0 ||
    filters.entitySearch.trim() !== "" ||
    filters.textSearch.trim() !== "" ||
    filters.timeMin !== "" ||
    filters.timeMax !== "";

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Event Log ({filtered.length}
          {hasActiveFilters ? ` / ${eventLog.length}` : ""})
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

      {/* Filter toolbar */}
      <div className="flex flex-wrap items-center gap-2 px-3 py-1.5 border-b border-gray-800 bg-gray-900/50">
        {/* Event type multi-select dropdown */}
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setTypeDropdownOpen((v) => !v)}
            className={`px-2 py-0.5 text-xs rounded border ${
              filters.selectedTypes.size > 0
                ? "border-blue-500/50 text-blue-400 bg-blue-500/10"
                : "border-gray-700 text-gray-400 bg-gray-800/50"
            } hover:border-gray-600 transition-colors`}
          >
            Type{filters.selectedTypes.size > 0 ? ` (${filters.selectedTypes.size})` : ""}
            <span className="ml-1 text-[10px]">{typeDropdownOpen ? "\u25B2" : "\u25BC"}</span>
          </button>
          {typeDropdownOpen && (
            <div className="absolute z-50 top-full left-0 mt-1 w-56 max-h-60 overflow-y-auto bg-gray-900 border border-gray-700 rounded shadow-lg">
              {eventTypes.length === 0 && (
                <div className="px-3 py-2 text-xs text-gray-500">No event types</div>
              )}
              {eventTypes.map((type) => (
                <label
                  key={type}
                  className="flex items-center gap-2 px-3 py-1 text-xs text-gray-300 hover:bg-gray-800 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={filters.selectedTypes.has(type)}
                    onChange={() => toggleType(type)}
                    className="rounded"
                  />
                  <span className="truncate">{type}</span>
                </label>
              ))}
              {filters.selectedTypes.size > 0 && (
                <button
                  onClick={() =>
                    setFilters((prev) => ({ ...prev, selectedTypes: new Set<string>() }))
                  }
                  className="w-full px-3 py-1 text-xs text-gray-500 hover:text-gray-300 hover:bg-gray-800 border-t border-gray-700 text-left"
                >
                  Clear selection
                </button>
              )}
            </div>
          )}
        </div>

        {/* Entity filter */}
        <input
          type="text"
          placeholder="Entity..."
          value={filters.entitySearch}
          onChange={(e) =>
            setFilters((prev) => ({ ...prev, entitySearch: e.target.value }))
          }
          className="w-24 px-2 py-0.5 text-xs rounded border border-gray-700 bg-gray-800/50 text-gray-300 placeholder-gray-600 focus:border-gray-500 focus:outline-none"
        />

        {/* Text search */}
        <input
          type="text"
          placeholder="Search..."
          value={filters.textSearch}
          onChange={(e) =>
            setFilters((prev) => ({ ...prev, textSearch: e.target.value }))
          }
          className="w-28 px-2 py-0.5 text-xs rounded border border-gray-700 bg-gray-800/50 text-gray-300 placeholder-gray-600 focus:border-gray-500 focus:outline-none"
        />

        {/* Time range */}
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <span>t:</span>
          <input
            type="number"
            placeholder="min"
            value={filters.timeMin}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, timeMin: e.target.value }))
            }
            className="w-16 px-1.5 py-0.5 text-xs rounded border border-gray-700 bg-gray-800/50 text-gray-300 placeholder-gray-600 focus:border-gray-500 focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <span>-</span>
          <input
            type="number"
            placeholder="max"
            value={filters.timeMax}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, timeMax: e.target.value }))
            }
            className="w-16 px-1.5 py-0.5 text-xs rounded border border-gray-700 bg-gray-800/50 text-gray-300 placeholder-gray-600 focus:border-gray-500 focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
        </div>

        {/* Auto-scroll toggle */}
        <label className="flex items-center gap-1 text-xs text-gray-500 cursor-pointer ml-auto">
          <input
            type="checkbox"
            checked={filters.autoScroll}
            onChange={(e) =>
              setFilters((prev) => ({ ...prev, autoScroll: e.target.checked }))
            }
            className="rounded"
          />
          Auto-scroll
        </label>

        {/* Clear filters */}
        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="px-2 py-0.5 text-xs rounded border border-gray-700 text-gray-500 hover:text-gray-300 hover:border-gray-600 bg-gray-800/50 transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Event list */}
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
                  {e.context ? (isExpanded ? "\u25BC" : "\u25B6") : " "}
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
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard
      .writeText(JSON.stringify(context, null, 2))
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {
        // Silently fail if clipboard access is denied
      });
  }, [context]);

  const trace = context.trace as { spans?: unknown[] } | undefined;
  const spans = trace?.spans;
  const otherKeys = Object.keys(context).filter((k) => {
    if (k === "trace") return false;
    const v = context[k];
    if (Array.isArray(v) && v.length === 0) return false;
    if (typeof v === "object" && v !== null && !Array.isArray(v) && Object.keys(v).length === 0) return false;
    return true;
  });

  return (
    <div className="px-4 py-2 bg-gray-900/80 border-b border-gray-800 space-y-2">
      {/* Copy button */}
      <div className="flex justify-end">
        <button
          onClick={handleCopy}
          className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${
            copied
              ? "border-emerald-600 text-emerald-400 bg-emerald-500/10"
              : "border-gray-700 text-gray-500 hover:text-gray-300 hover:border-gray-600 bg-gray-800/50"
          }`}
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
  if (v === null || v === undefined) return "\u2014";
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
