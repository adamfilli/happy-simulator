import {
  useEffect,
  useRef,
  useCallback,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";

interface Props {
  times: number[];
  values: number[];
  label: string;
  color?: string;
  yLabel?: string;
  xLabel?: string;
  yMin?: number | null;
  yMax?: number | null;
  onTimeRangeSelect?: (start: number, end: number) => void;
}

export interface TimeSeriesChartHandle {
  getCanvas: () => HTMLCanvasElement | null;
}

const TimeSeriesChart = forwardRef<TimeSeriesChartHandle, Props>(
  function TimeSeriesChart(
    {
      times,
      values,
      label,
      color = "#3b82f6",
      yLabel,
      xLabel,
      yMin: fixedYMin,
      yMax: fixedYMax,
      onTimeRangeSelect,
    },
    ref,
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [size, setSize] = useState<{ w: number; h: number } | null>(null);

    // Interaction state
    const [hoverX, setHoverX] = useState<number | null>(null);
    const [zoomRange, setZoomRange] = useState<[number, number] | null>(null);
    const [dragStart, setDragStart] = useState<number | null>(null);
    const [dragCurrent, setDragCurrent] = useState<number | null>(null);

    // Expose canvas element to parent via ref
    useImperativeHandle(ref, () => ({
      getCanvas: () => canvasRef.current,
    }));

    // Observe container size changes
    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      const ro = new ResizeObserver((entries) => {
        const entry = entries[0];
        if (!entry) return;
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setSize({ w: width, h: height });
        }
      });
      ro.observe(container);
      return () => ro.disconnect();
    }, []);

    // Binary search: find index of the closest time to `target`
    const findClosestIndex = useCallback(
      (target: number): number => {
        if (times.length === 0) return -1;
        let lo = 0;
        let hi = times.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (times[mid] < target) lo = mid + 1;
          else hi = mid;
        }
        // lo is the first index >= target; compare with lo-1 to find closest
        if (lo > 0 && Math.abs(times[lo - 1] - target) <= Math.abs(times[lo] - target)) {
          return lo - 1;
        }
        return lo;
      },
      [times],
    );

    const draw = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas || !size || times.length === 0) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = size.w * dpr;
      canvas.height = size.h * dpr;
      ctx.scale(dpr, dpr);

      const w = size.w;
      const h = size.h;
      const hasYLabel = !!yLabel;
      const hasXLabel = !!xLabel;
      const pad = {
        top: 8,
        right: 12,
        bottom: hasXLabel ? 36 : 24,
        left: hasYLabel ? 54 : 40,
      };
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;

      ctx.clearRect(0, 0, w, h);

      // Compute time range (zoom-aware)
      const fullTMin = times[0];
      const fullTMax = times[times.length - 1];
      const tMin = zoomRange ? zoomRange[0] : fullTMin;
      const tMax = zoomRange ? zoomRange[1] : fullTMax;
      const tRange = tMax - tMin || 1;

      // Compute value range (recompute for visible data when zoomed)
      let visVMin: number;
      let visVMax: number;
      if (fixedYMin != null && fixedYMax != null) {
        visVMin = fixedYMin;
        visVMax = fixedYMax;
      } else {
        // Find min/max of values within the visible time range
        let autoMin = Infinity;
        let autoMax = -Infinity;
        for (let i = 0; i < times.length; i++) {
          if (times[i] >= tMin && times[i] <= tMax) {
            if (values[i] < autoMin) autoMin = values[i];
            if (values[i] > autoMax) autoMax = values[i];
          }
        }
        if (!isFinite(autoMin)) {
          autoMin = 0;
          autoMax = 1;
        }
        visVMin = fixedYMin != null ? fixedYMin : 0;
        visVMax = fixedYMax != null ? fixedYMax : Math.max(autoMax, 1);
      }
      const vMin = visVMin;
      const vMax = visVMax;
      const vRange = vMax - vMin || 1;

      const toX = (t: number) => pad.left + ((t - tMin) / tRange) * plotW;
      const toY = (v: number) => pad.top + plotH - ((v - vMin) / vRange) * plotH;
      const fromX = (px: number) => tMin + ((px - pad.left) / plotW) * tRange;

      // Grid lines + Y-axis tick labels
      ctx.strokeStyle = "#1f2937";
      ctx.lineWidth = 1;
      const yTicks = 4;
      for (let i = 0; i <= yTicks; i++) {
        const v = vMin + (vRange * i) / yTicks;
        const y = toY(v);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();

        ctx.fillStyle = "#6b7280";
        ctx.font = "10px monospace";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(Number.isInteger(v) ? String(v) : v.toFixed(1), pad.left - 4, y);
      }

      // X-axis tick labels
      ctx.fillStyle = "#6b7280";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const xTicks = Math.min(5, times.length);
      for (let i = 0; i < xTicks; i++) {
        const t = tMin + (tRange * i) / (xTicks - 1 || 1);
        ctx.fillText(t.toFixed(1) + "s", toX(t), pad.top + plotH + 4);
      }

      // Y-axis label (rotated 90deg)
      if (yLabel) {
        ctx.save();
        ctx.fillStyle = "#9ca3af";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.translate(10, pad.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
      }

      // X-axis label
      if (xLabel) {
        ctx.fillStyle = "#9ca3af";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(xLabel, pad.left + plotW / 2, h - 14);
      }

      // Clip to plot area for data rendering
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad.left, pad.top, plotW, plotH);
      ctx.clip();

      // Data line (render all points; clipping hides out-of-range segments)
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < times.length; i++) {
        const x = toX(times[i]);
        const y = toY(values[i]);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Fill under curve
      if (times.length > 0) {
        ctx.lineTo(toX(times[times.length - 1]), toY(vMin));
        ctx.lineTo(toX(times[0]), toY(vMin));
        ctx.closePath();
        ctx.fillStyle = color + "18";
        ctx.fill();
      }

      // Current value dot (last data point)
      if (times.length > 0) {
        const lastIdx = times.length - 1;
        const lx = toX(times[lastIdx]);
        const ly = toY(values[lastIdx]);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
        ctx.fill();
        // White inner ring for contrast
        ctx.fillStyle = "#111827";
        ctx.beginPath();
        ctx.arc(lx, ly, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore(); // remove clip

      // --- Interactive overlays (drawn after main chart, outside clip) ---

      // Drag selection rectangle
      if (dragStart != null && dragCurrent != null) {
        const x0 = Math.min(dragStart, dragCurrent);
        const x1 = Math.max(dragStart, dragCurrent);
        // Clamp to plot area
        const rx0 = Math.max(x0, pad.left);
        const rx1 = Math.min(x1, pad.left + plotW);
        if (rx1 > rx0) {
          ctx.fillStyle = "rgba(59, 130, 246, 0.15)";
          ctx.fillRect(rx0, pad.top, rx1 - rx0, plotH);
          ctx.strokeStyle = "rgba(59, 130, 246, 0.5)";
          ctx.lineWidth = 1;
          ctx.setLineDash([]);
          ctx.strokeRect(rx0, pad.top, rx1 - rx0, plotH);
        }
      }

      // Hover crosshair + tooltip
      if (hoverX != null && dragStart == null) {
        // Clamp hoverX to plot area
        const clampedX = Math.max(pad.left, Math.min(hoverX, pad.left + plotW));
        const hoverTime = fromX(clampedX);
        const nearestIdx = findClosestIndex(hoverTime);

        if (nearestIdx >= 0 && nearestIdx < times.length) {
          const snapTime = times[nearestIdx];
          const snapValue = values[nearestIdx];
          const snapX = toX(snapTime);
          const snapY = toY(snapValue);

          // Only show if the snapped point is within the visible plot area
          if (snapX >= pad.left - 1 && snapX <= pad.left + plotW + 1) {
            // Vertical dashed crosshair line
            ctx.save();
            ctx.strokeStyle = "#6b7280";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(snapX, pad.top);
            ctx.lineTo(snapX, pad.top + plotH);
            ctx.stroke();
            ctx.restore();

            // Dot on data line at snapped point
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(snapX, snapY, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = "#111827";
            ctx.beginPath();
            ctx.arc(snapX, snapY, 2, 0, Math.PI * 2);
            ctx.fill();

            // Tooltip box
            const tooltipText = `t = ${snapTime.toFixed(2)}s, value = ${snapValue.toFixed(2)}`;
            ctx.font = "10px monospace";
            const textMetrics = ctx.measureText(tooltipText);
            const tooltipW = textMetrics.width + 12;
            const tooltipH = 20;
            const tooltipPad = 6;

            // Position tooltip: prefer right of cursor, flip left if too close to right edge
            let tooltipX = snapX + 10;
            if (tooltipX + tooltipW > w - pad.right) {
              tooltipX = snapX - tooltipW - 10;
            }
            // Vertical position: above the dot, but keep within bounds
            let tooltipY = snapY - tooltipH - 8;
            if (tooltipY < pad.top) {
              tooltipY = snapY + 12;
            }

            // Background
            ctx.fillStyle = "rgba(17, 24, 39, 0.92)";
            ctx.beginPath();
            const r = 3;
            ctx.moveTo(tooltipX + r, tooltipY);
            ctx.lineTo(tooltipX + tooltipW - r, tooltipY);
            ctx.arcTo(tooltipX + tooltipW, tooltipY, tooltipX + tooltipW, tooltipY + r, r);
            ctx.lineTo(tooltipX + tooltipW, tooltipY + tooltipH - r);
            ctx.arcTo(
              tooltipX + tooltipW,
              tooltipY + tooltipH,
              tooltipX + tooltipW - r,
              tooltipY + tooltipH,
              r,
            );
            ctx.lineTo(tooltipX + r, tooltipY + tooltipH);
            ctx.arcTo(tooltipX, tooltipY + tooltipH, tooltipX, tooltipY + tooltipH - r, r);
            ctx.lineTo(tooltipX, tooltipY + r);
            ctx.arcTo(tooltipX, tooltipY, tooltipX + r, tooltipY, r);
            ctx.closePath();
            ctx.fill();

            // Border
            ctx.strokeStyle = "#374151";
            ctx.lineWidth = 1;
            ctx.setLineDash([]);
            ctx.stroke();

            // Text
            ctx.fillStyle = "#e5e7eb";
            ctx.font = "10px monospace";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(tooltipText, tooltipX + tooltipPad, tooltipY + tooltipH / 2);
          }
        }
      }

      // Zoom indicator: show a subtle label when zoomed
      if (zoomRange) {
        ctx.fillStyle = "#6b7280";
        ctx.font = "9px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "top";
        ctx.fillText("dbl-click to reset zoom", w - pad.right, pad.top + 2);
      }
    }, [
      times,
      values,
      color,
      size,
      yLabel,
      xLabel,
      fixedYMin,
      fixedYMax,
      hoverX,
      zoomRange,
      dragStart,
      dragCurrent,
      findClosestIndex,
    ]);

    useEffect(() => {
      draw();
    }, [draw]);

    // --- Mouse event handlers ---

    const getMouseX = useCallback(
      (e: React.MouseEvent<HTMLCanvasElement>): number => {
        const canvas = canvasRef.current;
        if (!canvas) return 0;
        const rect = canvas.getBoundingClientRect();
        return e.clientX - rect.left;
      },
      [],
    );

    const handleMouseMove = useCallback(
      (e: React.MouseEvent<HTMLCanvasElement>) => {
        const mx = getMouseX(e);
        setHoverX(mx);
        if (dragStart != null) {
          setDragCurrent(mx);
        }
      },
      [getMouseX, dragStart],
    );

    const handleMouseDown = useCallback(
      (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (e.button !== 0) return; // only left click
        const mx = getMouseX(e);
        setDragStart(mx);
        setDragCurrent(mx);
      },
      [getMouseX],
    );

    const handleMouseUp = useCallback(
      (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (dragStart == null || dragCurrent == null || !size) {
          setDragStart(null);
          setDragCurrent(null);
          return;
        }

        const mx = getMouseX(e);
        const dist = Math.abs(mx - dragStart);

        if (dist > 5) {
          // Compute time range from pixel range
          const hasYLabel_ = !!yLabel;
          const padLeft = hasYLabel_ ? 54 : 40;
          const padRight = 12;
          const plotW = size.w - padLeft - padRight;

          const fullTMin = times[0];
          const fullTMax = times[times.length - 1];
          const currentTMin = zoomRange ? zoomRange[0] : fullTMin;
          const currentTMax = zoomRange ? zoomRange[1] : fullTMax;
          const currentTRange = currentTMax - currentTMin || 1;

          const fromXLocal = (px: number) => currentTMin + ((px - padLeft) / plotW) * currentTRange;

          const t0 = fromXLocal(Math.min(dragStart, mx));
          const t1 = fromXLocal(Math.max(dragStart, mx));

          // Clamp to data bounds
          const clampedT0 = Math.max(t0, fullTMin);
          const clampedT1 = Math.min(t1, fullTMax);

          if (clampedT1 > clampedT0) {
            setZoomRange([clampedT0, clampedT1]);
            if (onTimeRangeSelect) {
              onTimeRangeSelect(clampedT0, clampedT1);
            }
          }
        }

        setDragStart(null);
        setDragCurrent(null);
      },
      [dragStart, dragCurrent, getMouseX, size, times, yLabel, zoomRange, onTimeRangeSelect],
    );

    const handleMouseLeave = useCallback(() => {
      setHoverX(null);
      if (dragStart != null) {
        setDragStart(null);
        setDragCurrent(null);
      }
    }, [dragStart]);

    const handleDoubleClick = useCallback(() => {
      setZoomRange(null);
      if (onTimeRangeSelect) {
        // Notify parent that zoom was reset (full range)
        if (times.length > 0) {
          onTimeRangeSelect(times[0], times[times.length - 1]);
        }
      }
    }, [onTimeRangeSelect, times]);

    return (
      <div className="flex flex-col h-full min-h-0">
        <span className="text-[10px] text-gray-500 uppercase tracking-wide px-1 mb-1 shrink-0">
          {label}
        </span>
        <div ref={containerRef} className="flex-1 min-h-0">
          <canvas
            ref={canvasRef}
            style={{ width: "100%", height: "100%", cursor: dragStart != null ? "col-resize" : "crosshair" }}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            onDoubleClick={handleDoubleClick}
          />
        </div>
      </div>
    );
  },
);

export default TimeSeriesChart;
