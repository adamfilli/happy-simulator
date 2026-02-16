import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";

type GroupNodeData = {
  label: string;
  entityType: string;
  category: string;
  color: string;
  memberCount: number;
  selected?: boolean;
};

type GroupNodeType = Node<GroupNodeData, "group">;

function GroupNode({ data }: NodeProps<GroupNodeType>) {
  const { label, entityType, color, memberCount, selected } = data;

  return (
    <>
      <Handle type="target" position={Position.Left} id="left" className="!bg-gray-600 !w-2 !h-2" />
      <Handle type="target" position={Position.Top} id="top" className="!bg-cyan-700 !w-2 !h-2" />
      <div className="relative" style={{ width: 206, height: 96, overflow: "visible" }}>
        {/* Stacked card behind (deepest) */}
        <div
          className="absolute rounded-lg border-2 pointer-events-none"
          style={{
            width: 200,
            height: 90,
            top: 6,
            left: 6,
            backgroundColor: "#0d1117",
            borderColor: `${color}20`,
          }}
        />
        {/* Stacked card behind (middle) */}
        <div
          className="absolute rounded-lg border-2 pointer-events-none"
          style={{
            width: 200,
            height: 90,
            top: 3,
            left: 3,
            backgroundColor: "#111520",
            borderColor: `${color}30`,
          }}
        />
        {/* Main card */}
        <div
          className="absolute rounded-lg px-3 py-2 border-2 transition-all"
          style={{
            width: 200,
            height: 90,
            top: 0,
            left: 0,
            backgroundColor: "#111827",
            borderColor: selected ? color : `${color}60`,
            boxShadow: selected ? `0 0 16px ${color}50` : `0 0 6px ${color}15`,
          }}
        >
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold text-white truncate">{label}</div>
            <span
              className="ml-1 px-1.5 py-0.5 text-[10px] font-bold rounded-full"
              style={{
                backgroundColor: `${color}25`,
                color: color,
              }}
            >
              {memberCount >= 1000
                ? `${(memberCount / 1000).toFixed(memberCount >= 10000 ? 0 : 1)}k`
                : memberCount.toLocaleString()}
            </span>
          </div>
          <div className="text-[10px] mt-0.5" style={{ color: `${color}cc` }}>
            {entityType}
          </div>
          <div className="mt-2 text-[10px] text-gray-500">
            Click to inspect members
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} id="right" className="!bg-gray-600 !w-2 !h-2" />
      <Handle type="source" position={Position.Bottom} id="bottom" className="!bg-cyan-700 !w-2 !h-2" />
    </>
  );
}

export default memo(GroupNode);
