import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";

type EntityNodeData = {
  label: string;
  entityType: string;
  category: string;
  color: string;
  metrics: Record<string, unknown>;
  selected?: boolean;
};

type EntityNodeType = Node<EntityNodeData, "entity">;

function EntityNode({ data }: NodeProps<EntityNodeType>) {
  const { label, entityType, color, metrics, selected } = data;

  // Pick top 2 metrics to show on the node
  const metricEntries = Object.entries(metrics).slice(0, 2);

  return (
    <>
      <Handle type="target" position={Position.Left} className="!bg-gray-600 !w-2 !h-2" />
      <div
        className="rounded-lg px-3 py-2 min-w-[140px] border-2 transition-all"
        style={{
          backgroundColor: "#111827",
          borderColor: selected ? color : `${color}40`,
          boxShadow: selected ? `0 0 12px ${color}40` : "none",
        }}
      >
        <div className="text-xs font-semibold text-white truncate">{label}</div>
        <div className="text-[10px] mt-0.5" style={{ color: `${color}cc` }}>
          {entityType}
        </div>
        {metricEntries.length > 0 && (
          <div className="mt-1.5 space-y-0.5">
            {metricEntries.map(([key, val]) => (
              <div key={key} className="flex justify-between text-[10px]">
                <span className="text-gray-500">{key}</span>
                <span className="text-gray-300 font-mono">
                  {typeof val === "number"
                    ? Number.isInteger(val)
                      ? val
                      : (val as number).toFixed(3)
                    : String(val)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-gray-600 !w-2 !h-2" />
    </>
  );
}

export default memo(EntityNode);
