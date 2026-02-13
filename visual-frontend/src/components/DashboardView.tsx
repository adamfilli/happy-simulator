import { useState, useEffect } from "react";
import { useSimStore } from "../hooks/useSimState";
import DashboardPanel from "./DashboardPanel";

interface ProbeInfo {
  name: string;
  target: string;
  metric: string;
}

export default function DashboardView() {
  const panels = useSimStore((s) => s.dashboardPanels);
  const addDashboardPanel = useSimStore((s) => s.addDashboardPanel);
  const [showDropdown, setShowDropdown] = useState(false);
  const [probes, setProbes] = useState<ProbeInfo[]>([]);

  useEffect(() => {
    if (!showDropdown) return;
    fetch("/api/probes")
      .then((r) => r.json())
      .then((data: ProbeInfo[]) => setProbes(data))
      .catch(() => setProbes([]));
  }, [showDropdown]);

  const handleAddProbe = (probe: ProbeInfo) => {
    // Avoid duplicates
    if (panels.some((p) => p.probeName === probe.name)) {
      setShowDropdown(false);
      return;
    }
    const stagger = panels.length * 30;
    addDashboardPanel({
      id: crypto.randomUUID(),
      probeName: probe.name,
      label: `${probe.target}.${probe.metric}`,
      x: 20 + stagger,
      y: 20 + stagger,
    });
    setShowDropdown(false);
  };

  return (
    <div className="w-full h-full relative overflow-auto bg-gray-950">
      {/* Panels */}
      {panels.map((panel) => (
        <DashboardPanel
          key={panel.id}
          id={panel.id}
          probeName={panel.probeName}
          label={panel.label}
          x={panel.x}
          y={panel.y}
        />
      ))}

      {/* Empty state */}
      {panels.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="text-sm text-gray-600">
            Add charts from the button below, or pin probes from the Inspector.
          </span>
        </div>
      )}

      {/* Add Chart button */}
      <div className="absolute bottom-4 right-4">
        <div className="relative">
          <button
            onClick={() => setShowDropdown(!showDropdown)}
            className="px-3 py-1.5 text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white rounded shadow"
          >
            + Add Chart
          </button>
          {showDropdown && (
            <div className="absolute bottom-full right-0 mb-1 w-56 bg-gray-800 border border-gray-700 rounded shadow-lg max-h-60 overflow-y-auto">
              {probes.length === 0 ? (
                <div className="px-3 py-2 text-xs text-gray-500">No probes available</div>
              ) : (
                probes.map((probe) => {
                  const already = panels.some((p) => p.probeName === probe.name);
                  return (
                    <button
                      key={probe.name}
                      onClick={() => handleAddProbe(probe)}
                      disabled={already}
                      className={`w-full text-left px-3 py-1.5 text-xs ${
                        already
                          ? "text-gray-600 cursor-default"
                          : "text-gray-300 hover:bg-gray-700"
                      }`}
                    >
                      {probe.target}.{probe.metric}
                      {already && <span className="ml-1 text-gray-600">(added)</span>}
                    </button>
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
