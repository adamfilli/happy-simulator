import { useEffect, useRef } from "react";

interface Props {
  times: number[];
  values: number[];
  label: string;
  color?: string;
}

export default function TimeSeriesChart({ times, values, label, color = "#3b82f6" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || times.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 8, right: 12, bottom: 24, left: 40 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Compute ranges
    const tMin = times[0];
    const tMax = times[times.length - 1];
    const tRange = tMax - tMin || 1;
    const vMin = 0;
    const vMax = Math.max(...values, 1);
    const vRange = vMax - vMin || 1;

    const toX = (t: number) => pad.left + ((t - tMin) / tRange) * plotW;
    const toY = (v: number) => pad.top + plotH - ((v - vMin) / vRange) * plotH;

    // Grid lines
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

      // Y-axis labels
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(Number.isInteger(v) ? String(v) : v.toFixed(1), pad.left - 4, y);
    }

    // X-axis labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTicks = Math.min(5, times.length);
    for (let i = 0; i < xTicks; i++) {
      const idx = Math.floor((i / (xTicks - 1 || 1)) * (times.length - 1));
      const t = times[idx];
      ctx.fillText(t.toFixed(1) + "s", toX(t), h - pad.bottom + 4);
    }

    // Data line
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < times.length; i++) {
      const x = toX(times[i]);
      const y = toY(values[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under curve
    ctx.lineTo(toX(times[times.length - 1]), toY(vMin));
    ctx.lineTo(toX(times[0]), toY(vMin));
    ctx.closePath();
    ctx.fillStyle = color + "18";
    ctx.fill();
  }, [times, values, color]);

  return (
    <div className="flex flex-col">
      <span className="text-[10px] text-gray-500 uppercase tracking-wide px-1 mb-1">
        {label}
      </span>
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: 140 }}
      />
    </div>
  );
}
