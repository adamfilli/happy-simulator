import { useEffect, useRef, useCallback, useState } from "react";

interface Props {
  currentTime: number;
  endTime: number | null;
  breakpoints?: Array<{ id: string; type: string; time_s?: number }>;
  onSeekTo: (time_s: number) => void;
}

const BAR_HEIGHT = 24;
const PLAYHEAD_TRIANGLE_SIZE = 5;
const DIAMOND_SIZE = 6;
const LABEL_FONT = "9px monospace";
const LABEL_COLOR = "#6b7280";
const BG_COLOR = "#111827";
const ELAPSED_COLOR = "#1f2937";
const PLAYHEAD_COLOR = "#ffffff";
const BREAKPOINT_COLOR = "#f59e0b";

export default function Timeline({ currentTime, endTime, breakpoints, onSeekTo }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number>(0);

  // Observe container width changes
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width: w } = entry.contentRect;
      if (w > 0) {
        setWidth(w);
      }
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  const getMaxTime = useCallback((): number => {
    if (endTime != null && endTime > 0) {
      return endTime;
    }
    // Infinite sim: auto-extend beyond current time
    return currentTime > 0 ? currentTime * 1.5 : 10;
  }, [currentTime, endTime]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const h = BAR_HEIGHT;
    canvas.width = width * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const maxTime = getMaxTime();

    // Background
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, width, h);

    // Elapsed portion
    if (currentTime > 0 && maxTime > 0) {
      const elapsedWidth = Math.min((currentTime / maxTime) * width, width);
      ctx.fillStyle = ELAPSED_COLOR;
      ctx.fillRect(0, 0, elapsedWidth, h);
    }

    // Time labels along the bottom edge
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = LABEL_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    const labelCount = Math.max(2, Math.floor(width / 80));
    for (let i = 0; i <= labelCount; i++) {
      const t = (maxTime * i) / labelCount;
      const x = (i / labelCount) * width;

      // Format the label
      let label: string;
      if (t >= 3600) {
        label = (t / 3600).toFixed(1) + "h";
      } else if (t >= 60) {
        label = (t / 60).toFixed(1) + "m";
      } else if (maxTime >= 100) {
        label = Math.round(t) + "s";
      } else {
        label = t.toFixed(1) + "s";
      }

      // Nudge first and last labels inward so they don't clip
      if (i === 0) {
        ctx.textAlign = "left";
      } else if (i === labelCount) {
        ctx.textAlign = "right";
      } else {
        ctx.textAlign = "center";
      }

      ctx.fillText(label, x, h - 1);
    }

    // Breakpoint markers (diamonds)
    if (breakpoints) {
      for (const bp of breakpoints) {
        if (bp.time_s == null || bp.time_s < 0) continue;
        if (bp.time_s > maxTime) continue;

        const bpX = (bp.time_s / maxTime) * width;
        const cy = h / 2;
        const halfSize = DIAMOND_SIZE / 2;

        ctx.fillStyle = BREAKPOINT_COLOR;
        ctx.beginPath();
        ctx.moveTo(bpX, cy - halfSize);       // top
        ctx.lineTo(bpX + halfSize, cy);       // right
        ctx.lineTo(bpX, cy + halfSize);       // bottom
        ctx.lineTo(bpX - halfSize, cy);       // left
        ctx.closePath();
        ctx.fill();
      }
    }

    // Playhead: vertical line + triangle handle at top
    if (maxTime > 0) {
      const playheadX = Math.min((currentTime / maxTime) * width, width);

      // Vertical line
      ctx.strokeStyle = PLAYHEAD_COLOR;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, h);
      ctx.stroke();

      // Downward-pointing triangle at the top
      ctx.fillStyle = PLAYHEAD_COLOR;
      ctx.beginPath();
      ctx.moveTo(playheadX - PLAYHEAD_TRIANGLE_SIZE, 0);
      ctx.lineTo(playheadX + PLAYHEAD_TRIANGLE_SIZE, 0);
      ctx.lineTo(playheadX, PLAYHEAD_TRIANGLE_SIZE * 1.5);
      ctx.closePath();
      ctx.fill();
    }
  }, [width, currentTime, breakpoints, getMaxTime]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const canvasWidth = rect.width;

      if (clickX < 0 || clickX > canvasWidth) return;

      const maxTime = getMaxTime();
      const clickedTime = (clickX / canvasWidth) * maxTime;

      // Only seek forward (time > currentTime)
      if (clickedTime > currentTime) {
        onSeekTo(clickedTime);
      }
    },
    [currentTime, getMaxTime, onSeekTo],
  );

  return (
    <div
      ref={containerRef}
      className="w-full border-b border-gray-800"
      style={{ height: BAR_HEIGHT }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", cursor: "pointer" }}
        onClick={handleClick}
      />
    </div>
  );
}
