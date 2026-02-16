import { type EdgeProps, getSmoothStepPath } from "@xyflow/react";
import { useRef } from "react";
import { useSimStore } from "../hooks/useSimState";

interface AnimatedEdgeData {
  isProbe?: boolean;
  statsKey?: string;
}

/** Snap duration to 0.5s steps so the animation doesn't restart on tiny rate changes. */
function quantize(val: number, step: number): number {
  return Math.round(val / step) * step;
}

export default function AnimatedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps) {
  const edgeData = (data || {}) as AnimatedEdgeData;
  const isProbe = edgeData.isProbe ?? false;
  const statsKey = edgeData.statsKey ?? "";
  const rate = useSimStore((s) => s.edgeStats[statsKey]?.rate ?? 0);

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const strokeColor = isProbe ? "#06b6d4" : "#4b5563";
  const strokeWidth = isProbe ? 1.5 : 2;

  const active = rate > 0 && !isProbe;

  // Quantize to 0.5s steps so the SVG animation doesn't constantly restart
  const rawDuration = rate > 0 ? Math.max(1.5, 6 / (1 + rate / 10)) : 0;
  const quantized = quantize(rawDuration, 0.5);
  const stableDuration = useRef(quantized);
  if (active && Math.abs(quantized - stableDuration.current) >= 0.5) {
    stableDuration.current = quantized;
  }

  return (
    <>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={isProbe ? "6 4" : undefined}
        markerEnd={markerEnd as string}
      />
      {active && (
        <circle r={3} fill="#22c55e">
          <animateMotion
            dur={`${stableDuration.current}s`}
            repeatCount="indefinite"
            path={edgePath}
          />
        </circle>
      )}
    </>
  );
}
