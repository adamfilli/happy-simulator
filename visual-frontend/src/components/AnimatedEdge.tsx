import { type EdgeProps, getSmoothStepPath } from "@xyflow/react";
import { useSimStore } from "../hooks/useSimState";

interface AnimatedEdgeData {
  isProbe?: boolean;
  statsKey?: string;
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

  // Animation duration inversely proportional to rate (faster = shorter duration)
  const animDuration = rate > 0 ? Math.max(0.3, 3 / (1 + rate / 5)) : 0;

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
      {rate > 0 && !isProbe && (
        <circle r={3} fill="#22c55e">
          <animateMotion
            dur={`${animDuration}s`}
            repeatCount="indefinite"
            path={edgePath}
          />
        </circle>
      )}
    </>
  );
}
