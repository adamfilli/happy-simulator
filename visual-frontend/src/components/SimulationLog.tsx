import { useRef, useEffect, useMemo } from "react";
import { useSimStore } from "../hooks/useSimState";

const LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] as const;

const LEVEL_RANK: Record<string, number> = {
  DEBUG: 0,
  INFO: 1,
  WARNING: 2,
  ERROR: 3,
  CRITICAL: 4,
};

const LEVEL_COLOR: Record<string, string> = {
  DEBUG: "text-gray-500",
  INFO: "text-gray-300",
  WARNING: "text-yellow-400",
  ERROR: "text-red-400",
  CRITICAL: "text-red-500 font-bold",
};

const ROW_BG: Record<string, string> = {
  WARNING: "bg-yellow-950/20",
  ERROR: "bg-red-950/20",
  CRITICAL: "bg-red-950/30",
};

export default function SimulationLog() {
  const simLogs = useSimStore((s) => s.simLogs);
  const logLevelFilter = useSimStore((s) => s.logLevelFilter);
  const setLogLevelFilter = useSimStore((s) => s.setLogLevelFilter);
  const endRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const minRank = LEVEL_RANK[logLevelFilter] ?? 0;
    return simLogs.filter((l) => (LEVEL_RANK[l.level] ?? 0) >= minRank);
  }, [simLogs, logLevelFilter]);

  const display = filtered.slice(-200);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [display.length]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Sim Log ({filtered.length})
        </span>
        <select
          value={logLevelFilter}
          onChange={(e) => setLogLevelFilter(e.target.value)}
          className="text-xs bg-gray-800 text-gray-300 border border-gray-700 rounded px-1.5 py-0.5"
        >
          {LOG_LEVELS.map((lvl) => (
            <option key={lvl} value={lvl}>
              {lvl}
            </option>
          ))}
        </select>
      </div>
      <div className="flex-1 overflow-y-auto text-xs font-mono">
        {display.map((l, i) => (
          <div
            key={i}
            className={`px-3 py-1 border-b border-gray-900 ${ROW_BG[l.level] ?? ""}`}
          >
            <div className="flex gap-2">
              <span className="text-gray-500 w-16 shrink-0 text-right">
                {l.time_s !== null ? l.time_s.toFixed(4) : "—"}
              </span>
              <span className={`w-12 shrink-0 ${LEVEL_COLOR[l.level] ?? "text-gray-300"}`}>
                {l.level.slice(0, 4)}
              </span>
              <span className="text-blue-400 truncate w-28 shrink-0">
                {l.logger_name}
              </span>
            </div>
            <div className={`pl-[7.5rem] ${LEVEL_COLOR[l.level] ?? "text-gray-300"}`}>
              {l.message}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
