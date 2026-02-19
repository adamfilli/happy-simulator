import { memo } from "react";

interface QueueWidgetProps {
  depth: number;
  color: string;
}

const MAX_VISUAL_ITEMS = 20;

function QueueWidget({ depth, color }: QueueWidgetProps) {
  const displayCount = Math.min(depth, MAX_VISUAL_ITEMS);
  const overflow = depth > MAX_VISUAL_ITEMS;

  return (
    <div className="flex flex-col items-center gap-1.5 w-full">
      <div
        className="flex items-center rounded border px-1 py-0.5 min-h-[18px] gap-px"
        style={{ borderColor: "#4b5563", backgroundColor: "#0a0f1a" }}
      >
        {displayCount > 0 ? (
          <>
            {Array.from({ length: displayCount }, (_, i) => (
              <div
                key={i}
                className="w-2.5 h-2.5 rounded-sm transition-opacity duration-200"
                style={{
                  backgroundColor: color,
                  opacity: 0.6 + 0.4 * ((displayCount - i) / displayCount),
                }}
              />
            ))}
            {overflow && (
              <span className="text-[9px] text-gray-500 ml-0.5">&hellip;</span>
            )}
          </>
        ) : (
          <span className="text-[9px] text-gray-600 italic px-1">empty</span>
        )}
      </div>
      <div className="text-[10px] text-gray-400 font-mono">
        depth: {depth}
      </div>
    </div>
  );
}

export default memo(QueueWidget);
