import { useRef, useEffect } from "react";

interface MiniSparklineProps {
  values: number[];
  color?: string;
}

/**
 * Minimal canvas sparkline — line + subtle area fill, no axes or labels.
 * Auto-scales Y to data range, X fills available width.
 */
export default function MiniSparkline({
  values,
  color = "#3b82f6",
}: MiniSparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || values.length < 2) return;

    const observer = new ResizeObserver(() => draw());
    observer.observe(container);
    draw();

    function draw() {
      const rect = container!.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = rect.width;
      const h = 40;

      canvas!.width = w * dpr;
      canvas!.height = h * dpr;
      canvas!.style.width = `${w}px`;
      canvas!.style.height = `${h}px`;

      const ctx = canvas!.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr);

      let min = values[0];
      let max = values[0];
      for (const v of values) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      // Avoid division by zero when all values are identical
      const range = max - min || 1;
      const pad = 2;

      const toX = (i: number) => (i / (values.length - 1)) * w;
      const toY = (v: number) =>
        h - pad - ((v - min) / range) * (h - pad * 2);

      // Area fill
      ctx.beginPath();
      ctx.moveTo(toX(0), h);
      for (let i = 0; i < values.length; i++) {
        ctx.lineTo(toX(i), toY(values[i]));
      }
      ctx.lineTo(toX(values.length - 1), h);
      ctx.closePath();
      ctx.fillStyle = color + "1a"; // ~10% opacity
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(values[0]));
      for (let i = 1; i < values.length; i++) {
        ctx.lineTo(toX(i), toY(values[i]));
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    return () => observer.disconnect();
  }, [values, color]);

  if (values.length < 2) return null;

  return (
    <div ref={containerRef} className="w-full" style={{ height: 40 }}>
      <canvas ref={canvasRef} />
    </div>
  );
}
