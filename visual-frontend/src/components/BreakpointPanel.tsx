import { useState, useEffect, useCallback } from "react";

interface BreakpointInfo {
  id: string;
  type: "time" | "event_count" | "event_type" | "metric" | "custom";
  one_shot: boolean;
  time_s?: number;
  count?: number;
  event_type?: string;
  entity_name?: string;
  attribute?: string;
  operator?: string;
  threshold?: number;
  description?: string;
}

type BreakpointType = "time" | "event_count" | "event_type" | "metric";

const OPERATORS = [
  { value: "gt", label: ">" },
  { value: "ge", label: ">=" },
  { value: "lt", label: "<" },
  { value: "le", label: "<=" },
  { value: "eq", label: "==" },
  { value: "ne", label: "!=" },
];

function describeBreakpoint(bp: BreakpointInfo): string {
  switch (bp.type) {
    case "time":
      return `Time >= ${bp.time_s}s`;
    case "event_count":
      return `Event count >= ${bp.count}`;
    case "event_type":
      return `Event type = ${bp.event_type}`;
    case "metric": {
      const op = OPERATORS.find((o) => o.value === bp.operator);
      return `${bp.entity_name}.${bp.attribute} ${op?.label ?? bp.operator} ${bp.threshold}`;
    }
    case "custom":
      return bp.description ?? "Custom breakpoint";
    default:
      return "Unknown";
  }
}

function typeLabel(type: BreakpointInfo["type"]): string {
  switch (type) {
    case "time":
      return "Time";
    case "event_count":
      return "Count";
    case "event_type":
      return "Event";
    case "metric":
      return "Metric";
    case "custom":
      return "Custom";
    default:
      return type;
  }
}

interface Props {
  open: boolean;
  onClose: () => void;
  entityNames: string[];
}

export default function BreakpointPanel({ open, onClose, entityNames }: Props) {
  const [breakpoints, setBreakpoints] = useState<BreakpointInfo[]>([]);
  const [loading, setLoading] = useState(false);

  // Form state
  const [bpType, setBpType] = useState<BreakpointType>("time");
  const [oneShot, setOneShot] = useState(true);
  const [timeS, setTimeS] = useState("");
  const [count, setCount] = useState("");
  const [eventType, setEventType] = useState("");
  const [entityName, setEntityName] = useState("");
  const [attribute, setAttribute] = useState("");
  const [operator, setOperator] = useState("gt");
  const [threshold, setThreshold] = useState("");

  const fetchBreakpoints = useCallback(async () => {
    try {
      const res = await fetch("/api/breakpoints");
      const data: BreakpointInfo[] = await res.json();
      setBreakpoints(data);
    } catch {
      // silently ignore fetch errors
    }
  }, []);

  useEffect(() => {
    if (open) {
      fetchBreakpoints();
    }
  }, [open, fetchBreakpoints]);

  const handleAdd = async () => {
    const body: Record<string, unknown> = { type: bpType, one_shot: oneShot };

    switch (bpType) {
      case "time": {
        const val = parseFloat(timeS);
        if (isNaN(val) || val < 0) return;
        body.time_s = val;
        break;
      }
      case "event_count": {
        const val = parseInt(count, 10);
        if (isNaN(val) || val <= 0) return;
        body.count = val;
        break;
      }
      case "event_type": {
        if (!eventType.trim()) return;
        body.event_type = eventType.trim();
        break;
      }
      case "metric": {
        if (!entityName || !attribute.trim()) return;
        const val = parseFloat(threshold);
        if (isNaN(val)) return;
        body.entity_name = entityName;
        body.attribute = attribute.trim();
        body.operator = operator;
        body.threshold = val;
        break;
      }
    }

    setLoading(true);
    try {
      await fetch("/api/breakpoints", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      await fetchBreakpoints();
      // Reset form fields
      setTimeS("");
      setCount("");
      setEventType("");
      setAttribute("");
      setThreshold("");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (bpId: string) => {
    setLoading(true);
    try {
      await fetch(`/api/breakpoints/${encodeURIComponent(bpId)}`, {
        method: "DELETE",
      });
      await fetchBreakpoints();
    } finally {
      setLoading(false);
    }
  };

  const handleClearAll = async () => {
    setLoading(true);
    try {
      await fetch("/api/breakpoints", { method: "DELETE" });
      await fetchBreakpoints();
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 border-t-0 mx-0 text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Breakpoints
        </span>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 text-sm leading-none px-1"
          title="Close panel"
        >
          &times;
        </button>
      </div>

      <div className="flex gap-4 p-3">
        {/* Left: breakpoint list */}
        <div className="flex-1 min-w-0">
          {breakpoints.length === 0 ? (
            <div className="text-gray-500 py-2">No breakpoints set.</div>
          ) : (
            <>
              <table className="w-full text-left">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800">
                    <th className="pb-1 pr-2 font-medium">Type</th>
                    <th className="pb-1 pr-2 font-medium">Description</th>
                    <th className="pb-1 pr-2 font-medium text-center">One-shot</th>
                    <th className="pb-1 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {breakpoints.map((bp) => (
                    <tr
                      key={bp.id}
                      className="border-b border-gray-800/50 text-gray-300"
                    >
                      <td className="py-1 pr-2">
                        <span className="px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">
                          {typeLabel(bp.type)}
                        </span>
                      </td>
                      <td className="py-1 pr-2 font-mono text-gray-300">
                        {describeBreakpoint(bp)}
                      </td>
                      <td className="py-1 pr-2 text-center text-gray-500">
                        {bp.one_shot ? "yes" : "no"}
                      </td>
                      <td className="py-1 text-right">
                        <button
                          onClick={() => handleDelete(bp.id)}
                          disabled={loading}
                          className="px-1.5 py-0.5 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded disabled:opacity-40"
                          title="Delete breakpoint"
                        >
                          &times;
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-2">
                <button
                  onClick={handleClearAll}
                  disabled={loading}
                  className="px-2 py-0.5 text-xs text-gray-500 hover:text-gray-300 hover:bg-gray-800 rounded border border-gray-700 disabled:opacity-40"
                >
                  Clear All
                </button>
              </div>
            </>
          )}
        </div>

        {/* Separator */}
        <div className="w-px bg-gray-700 shrink-0" />

        {/* Right: add form */}
        <div className="w-72 shrink-0 space-y-2">
          <div className="text-gray-400 font-semibold uppercase tracking-wide mb-1">
            Add Breakpoint
          </div>

          {/* Type selector */}
          <div className="flex items-center gap-2">
            <label className="text-gray-500 w-14 shrink-0">Type</label>
            <select
              value={bpType}
              onChange={(e) => setBpType(e.target.value as BreakpointType)}
              className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 text-xs focus:border-gray-500 focus:outline-none"
            >
              <option value="time">Time</option>
              <option value="event_count">Event Count</option>
              <option value="event_type">Event Type</option>
              <option value="metric">Metric</option>
            </select>
          </div>

          {/* Type-specific fields */}
          {bpType === "time" && (
            <div className="flex items-center gap-2">
              <label className="text-gray-500 w-14 shrink-0">Time (s)</label>
              <input
                type="number"
                value={timeS}
                onChange={(e) => setTimeS(e.target.value)}
                placeholder="5.0"
                min="0"
                step="any"
                className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 font-mono text-xs placeholder-gray-600 focus:border-gray-500 focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
            </div>
          )}

          {bpType === "event_count" && (
            <div className="flex items-center gap-2">
              <label className="text-gray-500 w-14 shrink-0">Count</label>
              <input
                type="number"
                value={count}
                onChange={(e) => setCount(e.target.value)}
                placeholder="1000"
                min="1"
                step="1"
                className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 font-mono text-xs placeholder-gray-600 focus:border-gray-500 focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
            </div>
          )}

          {bpType === "event_type" && (
            <div className="flex items-center gap-2">
              <label className="text-gray-500 w-14 shrink-0">Type</label>
              <input
                type="text"
                value={eventType}
                onChange={(e) => setEventType(e.target.value)}
                placeholder="Timeout"
                className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 text-xs placeholder-gray-600 focus:border-gray-500 focus:outline-none"
              />
            </div>
          )}

          {bpType === "metric" && (
            <>
              <div className="flex items-center gap-2">
                <label className="text-gray-500 w-14 shrink-0">Entity</label>
                <select
                  value={entityName}
                  onChange={(e) => setEntityName(e.target.value)}
                  className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 text-xs focus:border-gray-500 focus:outline-none"
                >
                  <option value="">Select entity...</option>
                  {entityNames.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-gray-500 w-14 shrink-0">Attr</label>
                <input
                  type="text"
                  value={attribute}
                  onChange={(e) => setAttribute(e.target.value)}
                  placeholder="depth"
                  className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 text-xs placeholder-gray-600 focus:border-gray-500 focus:outline-none"
                />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-gray-500 w-14 shrink-0">Op</label>
                <select
                  value={operator}
                  onChange={(e) => setOperator(e.target.value)}
                  className="w-16 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 text-xs focus:border-gray-500 focus:outline-none"
                >
                  {OPERATORS.map((op) => (
                    <option key={op.value} value={op.value}>
                      {op.label}
                    </option>
                  ))}
                </select>
                <input
                  type="number"
                  value={threshold}
                  onChange={(e) => setThreshold(e.target.value)}
                  placeholder="100"
                  step="any"
                  className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-gray-300 font-mono text-xs placeholder-gray-600 focus:border-gray-500 focus:outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
              </div>
            </>
          )}

          {/* One-shot checkbox */}
          <div className="flex items-center gap-2">
            <label className="text-gray-500 w-14 shrink-0">Options</label>
            <label className="flex items-center gap-1.5 text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={oneShot}
                onChange={(e) => setOneShot(e.target.checked)}
                className="rounded"
              />
              One-shot
            </label>
          </div>

          {/* Add button */}
          <div className="flex items-center gap-2">
            <div className="w-14 shrink-0" />
            <button
              onClick={handleAdd}
              disabled={loading}
              className="px-3 py-1 bg-purple-700 hover:bg-purple-600 disabled:opacity-40 rounded text-xs font-medium text-white"
            >
              Add
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
