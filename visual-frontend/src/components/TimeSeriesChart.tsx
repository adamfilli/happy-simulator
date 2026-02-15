import { useEffect, useRef, useCallback, useState, forwardRef, useImperativeHandle } from "react";

export interface TimeSeriesChartHandle {
  getCanvas: () => HTMLCanvasElement | null;
}

interface Props {
  times: number[];
  values: number[];
  label: string;
  color?: string;
  yLabel?: string;
  xLabel?: string;
  yMin?: number | null;
  yMax?: number | null;
}

/** Binary search for nearest data point by time. */
function findNearest(times: number[], target: number): number {
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - target) < Math.abs(times[lo] - target)) {
    return lo - 1;
  }
  return lo;
}

function formatValue(v: number): string {
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

interface Layout {
  pad: { top: number; right: number; bottom: number; left: number };
  plotW: number;
  plotH: number;
  tMin: number;
  tMax: number;
  toX: (t: number) => number;
  fromX: (px: number) => number;
  toY: (v: number) => number;
}

const TimeSeriesChart = forwardRef<TimeSeriesChartHandle, Props>(function TimeSeriesChart(
  {
    times,
    values,
    label,
    color = "#3b82f6",
    yLabel,
    xLabel,
    yMin: fixedYMin,
    yMax: fixedYMax,
  },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  // Interaction refs (mutable, no re-render)
  const mousePos = useRef<{ x: number; y: number } | null>(null);
  const dragStartX = useRef<number | null>(null);
  const isDragging = useRef(false);
  const layoutRef = useRef<Layout | null>(null);
  const rafId = useRef(0);

  // Zoom is state because it changes what's rendered
  const [zoomRange, setZoomRange] = useState<{ tMin: number; tMax: number } | null>(null);

  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
  }));

  // Observe container size
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

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

    // Compute time range (zoom or full)
    const tMin = zoomRange ? zoomRange.tMin : times[0];
    const tMax = zoomRange ? zoomRange.tMax : times[times.length - 1];
    const tRange = tMax - tMin || 1;

    // Auto-scale Y to visible data when zoomed
    let autoMax = 1;
    if (fixedYMax == null) {
      if (zoomRange) {
        for (let i = 0; i < times.length; i++) {
          if (times[i] >= tMin && times[i] <= tMax && values[i] > autoMax) {
            autoMax = values[i];
          }
        }
      } else {
        for (let i = 0; i < values.length; i++) {
          if (values[i] > autoMax) autoMax = values[i];
        }
      }
    }

    const vMin = fixedYMin != null ? fixedYMin : 0;
    const vMax = fixedYMax != null ? fixedYMax : autoMax;
    const vRange = vMax - vMin || 1;

    const toX = (t: number) => pad.left + ((t - tMin) / tRange) * plotW;
    const toY = (v: number) => pad.top + plotH - ((v - vMin) / vRange) * plotH;
    const fromX = (px: number) => tMin + ((px - pad.left) / plotW) * tRange;

    layoutRef.current = { pad, plotW, plotH, tMin, tMax, toX, fromX, toY };

    // ── Grid + Y ticks ──
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

    // ── X ticks ──
    ctx.fillStyle = "#6b7280";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTicks = 5;
    for (let i = 0; i < xTicks; i++) {
      const t = tMin + (tRange * i) / (xTicks - 1 || 1);
      ctx.fillText(t.toFixed(1) + "s", toX(t), pad.top + plotH + 4);
    }

    // ── Axis labels ──
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
    if (xLabel) {
      ctx.fillStyle = "#9ca3af";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(xLabel, pad.left + plotW / 2, h - 14);
    }

    // ── Clip to plot area ──
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, plotW, plotH);
    ctx.clip();

    // ── Data line ──
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

    // ── Fill under curve ──
    if (times.length > 0) {
      ctx.lineTo(toX(times[times.length - 1]), toY(vMin));
      ctx.lineTo(toX(times[0]), toY(vMin));
      ctx.closePath();
      ctx.fillStyle = color + "18";
      ctx.fill();
    }

    // ── Current value dot (last data point) ──
    if (times.length > 0) {
      const li = times.length - 1;
      const lx = toX(times[li]);
      const ly = toY(values[li]);
      ctx.beginPath();
      ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // ── Hover crosshair + tooltip ──
    const mouse = mousePos.current;
    if (mouse && !isDragging.current) {
      const mx = mouse.x;
      const my = mouse.y;
      if (
        mx >= pad.left &&
        mx <= pad.left + plotW &&
        my >= pad.top &&
        my <= pad.top + plotH
      ) {
        const mouseT = fromX(mx);
        const idx = findNearest(times, mouseT);
        const snapX = toX(times[idx]);
        const snapY = toY(values[idx]);

        // Dashed crosshair
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = "#6b7280";
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        ctx.moveTo(snapX, pad.top);
        ctx.lineTo(snapX, pad.top + plotH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pad.left, snapY);
        ctx.lineTo(pad.left + plotW, snapY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Snap dot
        ctx.beginPath();
        ctx.arc(snapX, snapY, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Tooltip
        const timeStr = times[idx].toFixed(2) + "s";
        const valStr = formatValue(values[idx]);
        const tooltipText = `${timeStr}  ${valStr}`;
        ctx.font = "10px monospace";
        const tw = ctx.measureText(tooltipText).width;
        const tooltipW = tw + 12;
        const tooltipH = 20;
        let tx = snapX + 10;
        let ty = snapY - tooltipH - 8;
        if (tx + tooltipW > pad.left + plotW) tx = snapX - tooltipW - 10;
        if (ty < pad.top) ty = snapY + 10;

        roundRect(ctx, tx, ty, tooltipW, tooltipH, 4);
        ctx.fillStyle = "rgba(17, 24, 39, 0.92)";
        ctx.fill();
        ctx.strokeStyle = "#374151";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = "#e5e7eb";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(tooltipText, tx + 6, ty + tooltipH / 2);
      }
    }

    // ── Drag selection rectangle ──
    if (isDragging.current && dragStartX.current != null && mouse) {
      const x0 = Math.max(pad.left, Math.min(dragStartX.current, mouse.x));
      const x1 = Math.min(pad.left + plotW, Math.max(dragStartX.current, mouse.x));
      ctx.fillStyle = "rgba(59, 130, 246, 0.15)";
      ctx.fillRect(x0, pad.top, x1 - x0, plotH);
      ctx.strokeStyle = "rgba(59, 130, 246, 0.6)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x0, pad.top, x1 - x0, plotH);
      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [times, values, color, size, yLabel, xLabel, fixedYMin, fixedYMax, zoomRange]);

  // Draw on dependency change
  useEffect(() => {
    cancelAnimationFrame(rafId.current);
    draw();
  }, [draw]);

  // Cleanup pending rAF on unmount
  useEffect(() => () => cancelAnimationFrame(rafId.current), []);

  // Stable requestDraw via ref indirection
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const requestDraw = useCallback(() => {
    cancelAnimationFrame(rafId.current);
    rafId.current = requestAnimationFrame(() => drawRef.current());
  }, []);

  // ── Mouse handlers ──

  const getLocalX = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return e.clientX - rect.left;
  };
  const getLocalPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current) return;
      mousePos.current = getLocalPos(e);
      requestDraw();
    },
    [requestDraw],
  );

  const handleMouseLeave = useCallback(() => {
    mousePos.current = null;
    isDragging.current = false;
    dragStartX.current = null;
    requestDraw();
  }, [requestDraw]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (e.button !== 0 || !canvasRef.current) return;
      const x = getLocalX(e);
      const layout = layoutRef.current;
      if (!layout) return;
      if (x >= layout.pad.left && x <= layout.pad.left + layout.plotW) {
        isDragging.current = true;
        dragStartX.current = x;
      }
    },
    [],
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDragging.current || dragStartX.current == null) {
        isDragging.current = false;
        return;
      }
      if (!canvasRef.current) return;
      const endX = getLocalX(e);
      const startX = dragStartX.current;
      isDragging.current = false;
      dragStartX.current = null;

      const layout = layoutRef.current;
      if (!layout || Math.abs(endX - startX) < 5) {
        requestDraw();
        return;
      }

      const left = Math.max(layout.pad.left, Math.min(startX, endX));
      const right = Math.min(layout.pad.left + layout.plotW, Math.max(startX, endX));
      setZoomRange({ tMin: layout.fromX(left), tMax: layout.fromX(right) });
    },
    [requestDraw],
  );

  const handleDoubleClick = useCallback(() => {
    setZoomRange(null);
  }, []);

  return (
    <div className="flex flex-col h-full min-h-0">
      <span className="text-[10px] text-gray-500 uppercase tracking-wide px-1 mb-1 shrink-0">
        {label}
      </span>
      <div ref={containerRef} className="flex-1 min-h-0">
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", cursor: "crosshair" }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onDoubleClick={handleDoubleClick}
        />
      </div>
    </div>
  );
});

export default TimeSeriesChart;
