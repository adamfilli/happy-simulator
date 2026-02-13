import { useState, useEffect } from "react";
import { GridLayout, useContainerWidth, verticalCompactor, type Layout } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import { useSimStore } from "../hooks/useSimState";
import DashboardPanel from "./DashboardPanel";
import type { ChartConfig } from "../types";

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
  const setActiveView = useSimStore((s) => s.setActiveView);
  const timeRange = useSimStore((s) => s.dashboardTimeRange);
  const setTimeRange = useSimStore((s) => s.setDashboardTimeRange);
  const simTime = useSimStore((s) => s.state?.time_s ?? 0);
  const [showDropdown, setShowDropdown] = useState(false);
  const [probes, setProbes] = useState<ProbeInfo[]>([]);
  const [chartsLoaded, setChartsLoaded] = useState(false);
  const [startInput, setStartInput] = useState("");
  const [endInput, setEndInput] = useState("");
  const { width, containerRef, mounted } = useContainerWidth();

  // Auto-load predefined charts on first mount
  useEffect(() => {
    if (chartsLoaded) return;
    setChartsLoaded(true);

    fetch("/api/charts")
      .then((r) => r.json())
      .then((configs: ChartConfig[]) => {
        if (configs.length === 0) return;
        // Only add charts that aren't already in the store
        const existing = useSimStore.getState().dashboardPanels;
        const existingChartIds = new Set(existing.map((p) => p.chartConfig?.chart_id));
        const newConfigs = configs.filter((c) => !existingChartIds.has(c.chart_id));
        if (newConfigs.length === 0) return;

        const offset = existing.length;
        for (let i = 0; i < newConfigs.length; i++) {
          const c = newConfigs[i];
          const idx = offset + i;
          addDashboardPanel({
            id: c.chart_id,
            label: c.title || `Chart ${c.chart_id}`,
            chartConfig: c,
            x: (idx % 3) * 4,
            y: Math.floor(idx / 3) * 4,
            w: 4,
            h: 4,
          });
        }
        setActiveView("dashboard");
      })
      .catch(() => {});
  }, [chartsLoaded, addDashboardPanel, setActiveView]);

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

  const applyTimeRange = () => {
    setTimeRange({
      start: startInput ? parseFloat(startInput) : null,
      end: endInput ? parseFloat(endInput) : null,
    });
  };

  const resetTimeRange = () => {
    setStartInput("");
    setEndInput("");
    setTimeRange({ start: null, end: null });
  };

  const hasTimeFilter = timeRange.start != null || timeRange.end != null;

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-auto bg-gray-950 flex flex-col">
      {/* Time range toolbar */}
      {panels.length > 0 && (
        <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-900 border-b border-gray-800 shrink-0 z-10">
          <span className="text-[10px] text-gray-500 uppercase tracking-wide">Time Range</span>
          <input
            type="number"
            placeholder="Start (s)"
            value={startInput}
            onChange={(e) => setStartInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && applyTimeRange()}
            step="any"
            className="w-20 px-1.5 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300 placeholder-gray-600 focus:border-blue-500 focus:outline-none"
          />
          <span className="text-gray-600 text-xs">&ndash;</span>
          <input
            type="number"
            placeholder="End (s)"
            value={endInput}
            onChange={(e) => setEndInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && applyTimeRange()}
            step="any"
            className="w-20 px-1.5 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300 placeholder-gray-600 focus:border-blue-500 focus:outline-none"
          />
          <button
            onClick={applyTimeRange}
            className="px-2 py-0.5 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded"
          >
            Apply
          </button>
          {hasTimeFilter && (
            <button
              onClick={resetTimeRange}
              className="px-2 py-0.5 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded"
            >
              All
            </button>
          )}
          <span className="text-[10px] text-gray-600 ml-1">
            sim: {simTime.toFixed(1)}s
            {hasTimeFilter && ` | showing ${timeRange.start ?? 0}s – ${timeRange.end ?? simTime.toFixed(1)}s`}
          </span>
        </div>
      )}

      {/* Grid content */}
      <div className="flex-1 relative overflow-auto">
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
                chartConfig={panel.chartConfig}
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

      </div>

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
