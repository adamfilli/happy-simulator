import { useRef, useEffect } from "react";
import { useSimStore } from "../hooks/useSimState";

export default function EventLog() {
  const eventLog = useSimStore((s) => s.eventLog);
  const showInternal = useSimStore((s) => s.showInternal);
  const toggleInternal = useSimStore((s) => s.toggleInternal);
  const endRef = useRef<HTMLDivElement>(null);

  const filtered = showInternal
    ? eventLog
    : eventLog.filter((e) => !e.is_internal);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [filtered.length]);

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
        {filtered.slice(-200).map((e, i) => (
          <div
            key={`${e.event_id}-${i}`}
            className={`px-3 py-1 border-b border-gray-900 flex gap-2 ${
              e.is_internal ? "text-gray-600" : "text-gray-300"
            }`}
          >
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
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
