import { useState, useEffect, useCallback } from "react";
import type { BreakpointInfo } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  entityNames: string[];
  onBreakpointsChanged: (breakpoints: BreakpointInfo[]) => void;
}

type BpType = "time" | "event_count" | "event_type" | "metric";

const TYPE_LABELS: Record<BpType, string> = {
  time: "Time",
  event_count: "Event Count",
  event_type: "Event Type",
  metric: "Metric",
};

const TYPE_COLORS: Record<string, string> = {
  time: "bg-amber-700",
  event_count: "bg-blue-700",
  event_type: "bg-emerald-700",
  metric: "bg-purple-700",
  custom: "bg-gray-600",
};

export default function BreakpointPanel({ open, onClose, entityNames, onBreakpointsChanged }: Props) {
  const [breakpoints, setBreakpoints] = useState<BreakpointInfo[]>([]);
  const [loading, setLoading] = useState(false);

  // Form state
  const [bpType, setBpType] = useState<BpType>("time");
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
      onBreakpointsChanged(data);
    } catch {
      // ignore fetch errors
    }
  }, [onBreakpointsChanged]);

  useEffect(() => {
    if (open) fetchBreakpoints();
  }, [open, fetchBreakpoints]);

  const handleAdd = async () => {
    const body: Record<string, unknown> = { type: bpType, one_shot: oneShot };
    if (bpType === "time") {
      const t = parseFloat(timeS);
      if (isNaN(t) || t < 0) return;
      body.time_s = t;
    } else if (bpType === "event_count") {
      const c = parseInt(count, 10);
      if (isNaN(c) || c < 1) return;
      body.count = c;
    } else if (bpType === "event_type") {
      if (!eventType.trim()) return;
      body.event_type = eventType.trim();
    } else if (bpType === "metric") {
      if (!entityName || !attribute.trim() || isNaN(parseFloat(threshold))) return;
      body.entity_name = entityName;
      body.attribute = attribute.trim();
      body.operator = operator;
      body.threshold = parseFloat(threshold);
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
    await fetch(`/api/breakpoints/${bpId}`, { method: "DELETE" });
    await fetchBreakpoints();
  };

  const handleClearAll = async () => {
    await fetch("/api/breakpoints", { method: "DELETE" });
    await fetchBreakpoints();
  };

  if (!open) return null;

  return (
    <div className="bg-gray-900 border-b border-gray-800 px-4 py-3 animate-in">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Breakpoints</h3>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 text-lg leading-none px-1"
          title="Close breakpoint panel"
        >
          &times;
        </button>
      </div>

      <div className="flex gap-4">
        {/* Left: breakpoint table */}
        <div className="flex-1 min-w-0">
          {breakpoints.length === 0 ? (
            <p className="text-xs text-gray-600 italic py-2">No breakpoints set.</p>
          ) : (
            <div className="max-h-40 overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800">
                    <th className="text-left py-1 pr-2 font-medium">Type</th>
                    <th className="text-left py-1 pr-2 font-medium">Description</th>
                    <th className="text-left py-1 pr-2 font-medium">One-shot</th>
                    <th className="text-right py-1 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {breakpoints.map((bp) => (
                    <tr key={bp.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="py-1 pr-2">
                        <span
                          className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${TYPE_COLORS[bp.type] || TYPE_COLORS.custom}`}
                        >
                          {bp.type}
                        </span>
                      </td>
                      <td className="py-1 pr-2 text-gray-300 font-mono truncate max-w-[200px]">
                        {bp.description || bp.id}
                      </td>
                      <td className="py-1 pr-2 text-gray-500">{bp.one_shot ? "yes" : "no"}</td>
                      <td className="py-1 text-right">
                        <button
                          onClick={() => handleDelete(bp.id)}
                          className="text-red-500 hover:text-red-400 text-[10px] font-medium px-1.5 py-0.5 rounded hover:bg-red-900/30"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {breakpoints.length > 0 && (
            <button
              onClick={handleClearAll}
              className="mt-2 text-[10px] text-red-500 hover:text-red-400 font-medium"
            >
              Clear All
            </button>
          )}
        </div>

        {/* Divider */}
        <div className="w-px bg-gray-800 shrink-0" />

        {/* Right: add form */}
        <div className="w-64 shrink-0 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <label className="text-[10px] text-gray-500 w-10 shrink-0">Type</label>
            <select
              value={bpType}
              onChange={(e) => setBpType(e.target.value as BpType)}
              className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-white"
            >
              {(Object.keys(TYPE_LABELS) as BpType[]).map((t) => (
                <option key={t} value={t}>{TYPE_LABELS[t]}</option>
              ))}
            </select>
          </div>

          {/* Type-specific fields */}
          {bpType === "time" && (
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-gray-500 w-10 shrink-0">Time</label>
              <input
                type="text"
                value={timeS}
                onChange={(e) => setTimeS(e.target.value)}
                placeholder="seconds"
                className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white placeholder-gray-600"
              />
            </div>
          )}

          {bpType === "event_count" && (
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-gray-500 w-10 shrink-0">Count</label>
              <input
                type="text"
                value={count}
                onChange={(e) => setCount(e.target.value)}
                placeholder="event count"
                className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white placeholder-gray-600"
              />
            </div>
          )}

          {bpType === "event_type" && (
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-gray-500 w-10 shrink-0">Type</label>
              <input
                type="text"
                value={eventType}
                onChange={(e) => setEventType(e.target.value)}
                placeholder="e.g. Request"
                className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white placeholder-gray-600"
              />
            </div>
          )}

          {bpType === "metric" && (
            <>
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-gray-500 w-10 shrink-0">Entity</label>
                <select
                  value={entityName}
                  onChange={(e) => setEntityName(e.target.value)}
                  className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-white"
                >
                  <option value="">Select entity...</option>
                  {entityNames.map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-gray-500 w-10 shrink-0">Attr</label>
                <input
                  type="text"
                  value={attribute}
                  onChange={(e) => setAttribute(e.target.value)}
                  placeholder="e.g. depth"
                  className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white placeholder-gray-600"
                />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-gray-500 w-10 shrink-0">Op</label>
                <select
                  value={operator}
                  onChange={(e) => setOperator(e.target.value)}
                  className="w-16 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-white"
                >
                  <option value="gt">&gt;</option>
                  <option value="ge">&gt;=</option>
                  <option value="lt">&lt;</option>
                  <option value="le">&lt;=</option>
                  <option value="eq">=</option>
                  <option value="ne">!=</option>
                </select>
                <input
                  type="text"
                  value={threshold}
                  onChange={(e) => setThreshold(e.target.value)}
                  placeholder="value"
                  className="flex-1 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white placeholder-gray-600"
                />
              </div>
            </>
          )}

          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-[10px] text-gray-500 cursor-pointer">
              <input
                type="checkbox"
                checked={oneShot}
                onChange={(e) => setOneShot(e.target.checked)}
                className="rounded border-gray-600 bg-gray-800 text-purple-500 focus:ring-0 focus:ring-offset-0"
              />
              One-shot
            </label>
            <div className="flex-1" />
            <button
              onClick={handleAdd}
              disabled={loading}
              className="px-3 py-1 bg-purple-700 hover:bg-purple-600 disabled:opacity-40 rounded text-xs font-medium"
            >
              Add
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
