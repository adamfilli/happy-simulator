import { useState, useEffect } from "react";
import { GridLayout, useContainerWidth, verticalCompactor, type Layout } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
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
  const updateDashboardLayout = useSimStore((s) => s.updateDashboardLayout);
  const removeDashboardPanel = useSimStore((s) => s.removeDashboardPanel);
  const [showDropdown, setShowDropdown] = useState(false);
  const [probes, setProbes] = useState<ProbeInfo[]>([]);
  const { width, containerRef, mounted } = useContainerWidth();

  useEffect(() => {
    if (!showDropdown) return;
    fetch("/api/probes")
      .then((r) => r.json())
      .then((data: ProbeInfo[]) => setProbes(data))
      .catch(() => setProbes([]));
  }, [showDropdown]);

  const handleAddProbe = (probe: ProbeInfo) => {
    if (panels.some((p) => p.probeName === probe.name)) {
      setShowDropdown(false);
      return;
    }
    const col = panels.length % 3;
    const row = Math.floor(panels.length / 3);
    addDashboardPanel({
      id: crypto.randomUUID(),
      probeName: probe.name,
      label: `${probe.target}.${probe.metric}`,
      x: col * 4,
      y: row * 4,
      w: 4,
      h: 4,
    });
    setShowDropdown(false);
  };

  const layout: Layout = panels.map((p) => ({
    i: p.id,
    x: p.x,
    y: p.y,
    w: p.w,
    h: p.h,
    minW: 2,
    minH: 2,
  }));

  const onLayoutChange = (newLayout: Layout) => {
    updateDashboardLayout(newLayout.map((item) => ({
      i: item.i,
      x: item.x,
      y: item.y,
      w: item.w,
      h: item.h,
    })));
  };

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-auto bg-gray-950">
      {panels.length > 0 && mounted ? (
        <GridLayout
          width={width}
          layout={layout}
          gridConfig={{ cols: 12, rowHeight: 50, margin: [12, 12] as const }}
          dragConfig={{ handle: ".drag-handle" }}
          compactor={verticalCompactor}
          onLayoutChange={onLayoutChange}
          autoSize
        >
          {panels.map((panel) => (
            <div key={panel.id}>
              <DashboardPanel
                id={panel.id}
                probeName={panel.probeName}
                label={panel.label}
                onClose={() => removeDashboardPanel(panel.id)}
              />
            </div>
          ))}
        </GridLayout>
      ) : panels.length === 0 ? (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="text-sm text-gray-600">
            Add charts from the button below, or pin probes from the Inspector.
          </span>
        </div>
      ) : null}

      {/* Add Chart button */}
      <div className="absolute bottom-4 right-4 z-50">
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
