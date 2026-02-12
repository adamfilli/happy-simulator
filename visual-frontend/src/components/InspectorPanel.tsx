import { useSimStore } from "../hooks/useSimState";

export default function InspectorPanel() {
  const state = useSimStore((s) => s.state);
  const selectedEntity = useSimStore((s) => s.selectedEntity);
  const topology = useSimStore((s) => s.topology);

  if (!state) return null;

  const entityName = selectedEntity;
  const entityData = entityName ? state.entities[entityName] : null;
  const nodeInfo = topology?.nodes.find((n) => n.id === entityName);

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
          Inspector
        </span>
      </div>

      {entityName && entityData ? (
        <div className="p-3 space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-white">{entityName}</h3>
            {nodeInfo && (
              <span className="text-xs text-gray-500">{nodeInfo.type}</span>
            )}
          </div>
          <div className="space-y-1">
            {Object.entries(entityData).map(([key, value]) => (
              <div key={key} className="flex justify-between text-xs">
                <span className="text-gray-400">{key}</span>
                <span className="text-white font-mono">
                  {formatValue(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="p-3 text-xs text-gray-500">
          Click a node to inspect its state
        </div>
      )}

      {/* Upcoming events */}
      {state.upcoming.length > 0 && (
        <div className="border-t border-gray-800 mt-auto">
          <div className="px-3 py-2">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
              Upcoming
            </span>
          </div>
          <div className="text-xs font-mono px-3 pb-2 space-y-1">
            {state.upcoming.slice(0, 8).map((e, i) => (
              <div key={i} className="flex gap-2 text-gray-400">
                <span className="w-14 text-right text-gray-500">
                  {e.time_s.toFixed(4)}
                </span>
                <span className="text-blue-400 truncate">{e.event_type}</span>
                <span className="text-gray-600">&rarr;</span>
                <span className="text-emerald-400 truncate">{e.target}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatValue(v: unknown): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  }
  if (typeof v === "object" && v !== null) {
    return JSON.stringify(v);
  }
  return String(v);
}
