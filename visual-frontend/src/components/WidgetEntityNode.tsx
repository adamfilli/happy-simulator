import { memo, useContext } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { CodePanelCtx } from "./CodePanelContext";
import SlotsWidget from "./widgets/SlotsWidget";
import QueueWidget from "./widgets/QueueWidget";

type WidgetNodeData = {
  label: string;
  entityType: string;
  category: string;
  color: string;
  metrics: Record<string, unknown>;
  selected?: boolean;
  widget: Record<string, unknown>;
};

type WidgetNodeType = Node<WidgetNodeData, "widget">;

export const WIDGET_SIZES: Record<string, { width: number; height: number }> = {
  queue: { width: 240, height: 90 },
  slots: { width: 200, height: 110 },
};
export const DEFAULT_WIDGET_SIZE = { width: 200, height: 100 };

function renderWidget(
  widget: Record<string, unknown>,
  metrics: Record<string, unknown>,
  color: string,
) {
  const widgetType = widget.type as string;

  if (widgetType === "slots") {
    const totalKey = widget.total as string;
    const activeKey = widget.active as string;
    const total = (metrics[totalKey] as number) ?? 0;
    const active = (metrics[activeKey] as number) ?? 0;
    return <SlotsWidget total={total} active={active} color={color} />;
  }

  if (widgetType === "queue") {
    const depthKey = widget.depth as string;
    const depth = (metrics[depthKey] as number) ?? 0;
    return <QueueWidget depth={depth} color={color} />;
  }

  return null;
}

function WidgetEntityNode({ data }: NodeProps<WidgetNodeType>) {
  const { label, entityType, color, metrics, selected, widget } = data;
  const ctx = useContext(CodePanelCtx);
  const onOpenCodePanel = ctx?.onOpenCodePanel;
  const hasCodePanel = ctx?.openPanels.has(label) ?? false;

  return (
    <>
      <Handle type="target" position={Position.Left} id="left" className="!bg-gray-600 !w-2 !h-2" />
      <Handle type="target" position={Position.Top} id="top" className="!bg-cyan-700 !w-2 !h-2" />
      <div
        className="rounded-lg px-3 py-2 min-w-[140px] transition-all"
        style={{
          backgroundColor: "#111827",
          border: `2px dashed ${selected ? color : `${color}40`}`,
          boxShadow: selected ? `0 0 12px ${color}40` : "none",
        }}
      >
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold text-white truncate">{label}</div>
          {onOpenCodePanel && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onOpenCodePanel(label);
              }}
              className={`ml-1 text-[10px] font-mono px-1 rounded transition-colors ${
                hasCodePanel
                  ? "text-amber-400 bg-amber-900/30"
                  : "text-gray-500 hover:text-amber-400 hover:bg-amber-900/20"
              }`}
              title={hasCodePanel ? "Code panel open" : "Open code panel"}
            >
              &lt;/&gt;
            </button>
          )}
        </div>
        <div className="text-[10px] mt-0.5" style={{ color: `${color}cc` }}>
          {entityType}
        </div>
        <div className="mt-2 rounded p-1.5" style={{ backgroundColor: "#0d1117" }}>
          {renderWidget(widget, metrics, color)}
        </div>
      </div>
      <Handle type="source" position={Position.Right} id="right" className="!bg-gray-600 !w-2 !h-2" />
      <Handle type="source" position={Position.Bottom} id="bottom" className="!bg-cyan-700 !w-2 !h-2" />
    </>
  );
}

export default memo(WidgetEntityNode);
