import { type EdgeProps, getSmoothStepPath } from "@xyflow/react";

interface AnimatedEdgeData {
  rate?: number;
  count?: number;
  isProbe?: boolean;
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
  const rate = edgeData.rate ?? 0;
  const isProbe = edgeData.isProbe ?? false;

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  // Dynamic styling based on rate
  const strokeWidth = isProbe ? 1.5 : Math.max(2, Math.min(6, 2 + rate / 5));
  const strokeColor = isProbe
    ? "#06b6d4"
    : rate > 20
      ? "#ef4444"  // red for high throughput
      : rate > 0
        ? "#22c55e"  // green for active
        : "#4b5563"; // gray for idle

  // Animation duration inversely proportional to rate (faster = shorter duration)
  const animDuration = rate > 0 ? Math.max(0.3, 3 / (1 + rate / 5)) : 0;

  return (
    <>
      {/* Base edge path */}
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={isProbe ? "6 4" : undefined}
        markerEnd={markerEnd as string}
      />
      {/* Animated particle when rate > 0 and not a probe */}
      {rate > 0 && !isProbe && (
        <>
          <circle r={3} fill={strokeColor}>
            <animateMotion
              dur={`${animDuration}s`}
              repeatCount="indefinite"
              path={edgePath}
            />
          </circle>
        </>
      )}
    </>
  );
}
