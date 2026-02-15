import { memo, useEffect, useRef, useState } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { useSimStore } from "../hooks/useSimState";
import type { CodeTrace } from "../types";

type CodePanelNodeData = {
  entityName: string;
  classname: string;
  methodName: string;
  sourceLines: string[];
  startLine: number;
  onClose: (entityName: string) => void;
};

type CodePanelNodeType = Node<CodePanelNodeData, "codePanel">;

const ANIMATION_INTERVAL_MS = 150;

function CodePanelNode({ data }: NodeProps<CodePanelNodeType>) {
  const { entityName, classname, methodName, sourceLines, startLine, onClose } = data;

  const codeTrace = useSimStore((s) => s.codeTraces.get(entityName));

  const [highlightedLine, setHighlightedLine] = useState<number | null>(null);
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevTraceRef = useRef<CodeTrace | undefined>(undefined);
  const codeContainerRef = useRef<HTMLDivElement>(null);

  // Animate code trace replay
  useEffect(() => {
    if (codeTrace && codeTrace !== prevTraceRef.current && codeTrace.lines.length > 0) {
      prevTraceRef.current = codeTrace;

      if (animRef.current) clearTimeout(animRef.current);

      let idx = 0;
      const animate = () => {
        if (idx < codeTrace.lines.length) {
          const line = codeTrace.lines[idx];
          setHighlightedLine(line.line_number);
          idx++;
          animRef.current = setTimeout(animate, ANIMATION_INTERVAL_MS);
        } else {
          animRef.current = setTimeout(() => {
            setHighlightedLine(null);
          }, 500);
        }
      };
      animate();
    }

    return () => {
      if (animRef.current) clearTimeout(animRef.current);
    };
  }, [codeTrace]);

  // Auto-scroll to highlighted line
  useEffect(() => {
    if (highlightedLine !== null && codeContainerRef.current) {
      const lineIdx = highlightedLine - startLine;
      const lineEl = codeContainerRef.current.children[lineIdx] as HTMLElement | undefined;
      if (lineEl) {
        lineEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }
  }, [highlightedLine, startLine]);

  return (
    <>
      <Handle type="target" position={Position.Left} id="code-left" className="!bg-amber-600 !w-2 !h-2" />
      <div
        className="flex flex-col bg-gray-950 border border-gray-700 rounded-lg overflow-hidden"
        style={{ width: "100%", height: "100%", minWidth: 300, minHeight: 200 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-gray-900 border-b border-gray-800 cursor-move drag-handle">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-amber-400 font-mono">&lt;/&gt;</span>
            <span className="text-gray-300 font-medium">{classname}</span>
            <span className="text-gray-600">.</span>
            <span className="text-blue-400">{methodName}</span>
          </div>
          <button
            onClick={() => onClose(entityName)}
            className="text-gray-500 hover:text-gray-300 text-xs px-1"
            title="Close code panel"
          >
            x
          </button>
        </div>

        {/* Source code */}
        <div ref={codeContainerRef} className="flex-1 overflow-auto font-mono text-[11px] leading-5">
          {sourceLines.map((line, i) => {
            const lineNo = startLine + i;
            const isHighlighted = highlightedLine === lineNo;

            return (
              <div
                key={lineNo}
                className={`flex items-stretch ${
                  isHighlighted ? "bg-amber-900/30" : "hover:bg-gray-900/50"
                }`}
              >
                {/* Line number */}
                <span className="w-10 flex-shrink-0 text-right pr-3 text-gray-600 select-none">
                  {lineNo}
                </span>
                {/* Code */}
                <pre className="flex-1 text-gray-300 whitespace-pre pr-4 overflow-hidden">
                  {line}
                </pre>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

export default memo(CodePanelNode);
