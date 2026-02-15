import { memo, useEffect, useRef, useState, useCallback } from "react";
import { Handle, Position, NodeResizer, type NodeProps, type Node } from "@xyflow/react";
import { useSimStore } from "../hooks/useSimState";
import type { CodeTrace, CodeBreakpointInfo } from "../types";

type CodePanelNodeData = {
  entityName: string;
  classname: string;
  methodName: string;
  sourceLines: string[];
  startLine: number;
  onClose: (entityName: string) => void;
  onToggleBreakpoint: (entityName: string, lineNumber: number) => void;
};

type CodePanelNodeType = Node<CodePanelNodeData, "codePanel">;

const ANIMATION_INTERVAL_MS = 150;

function CodePanelNode({ data }: NodeProps<CodePanelNodeType>) {
  const {
    entityName, classname, methodName, sourceLines, startLine,
    onClose, onToggleBreakpoint,
  } = data;

  const codeTrace = useSimStore((s) => s.codeTraces.get(entityName));
  const codePausedState = useSimStore((s) =>
    s.codePausedEntity === entityName ? s.codePausedState : null
  );
  const breakpoints = useSimStore((s) =>
    s.codeBreakpoints.filter((bp: CodeBreakpointInfo) => bp.entity_name === entityName)
  );

  const [highlightedLine, setHighlightedLine] = useState<number | null>(null);
  const [animLocals, setAnimLocals] = useState<Record<string, unknown> | null>(null);
  const [showLocals, setShowLocals] = useState(false);
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevTraceRef = useRef<CodeTrace | undefined>(undefined);
  const codeContainerRef = useRef<HTMLDivElement>(null);

  // Animate code trace replay
  useEffect(() => {
    if (codeTrace && codeTrace !== prevTraceRef.current && codeTrace.lines.length > 0) {
      prevTraceRef.current = codeTrace;

      // Cancel any existing animation
      if (animRef.current) clearTimeout(animRef.current);

      let idx = 0;
      const animate = () => {
        if (idx < codeTrace.lines.length) {
          const line = codeTrace.lines[idx];
          setHighlightedLine(line.line_number);
          if (line.locals) setAnimLocals(line.locals);
          idx++;
          animRef.current = setTimeout(animate, ANIMATION_INTERVAL_MS);
        } else {
          // Animation complete — keep last line highlighted briefly
          animRef.current = setTimeout(() => {
            if (!codePausedState) {
              setHighlightedLine(null);
              setAnimLocals(null);
            }
          }, 500);
        }
      };
      animate();
    }

    return () => {
      if (animRef.current) clearTimeout(animRef.current);
    };
  }, [codeTrace, codePausedState]);

  // When paused at a breakpoint, show that line
  useEffect(() => {
    if (codePausedState) {
      setHighlightedLine(codePausedState.line_number);
      setAnimLocals(codePausedState.locals);
      setShowLocals(true);
    }
  }, [codePausedState]);

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

  const breakpointLines = new Set(breakpoints.map((bp: CodeBreakpointInfo) => bp.line_number));

  const handleGutterClick = useCallback(
    (lineNumber: number) => {
      onToggleBreakpoint(entityName, lineNumber);
    },
    [entityName, onToggleBreakpoint]
  );

  return (
    <>
      <NodeResizer
        minWidth={300}
        minHeight={200}
        lineStyle={{ stroke: "#4b5563", strokeWidth: 1 }}
        handleStyle={{ width: 8, height: 8, background: "#6b7280", borderRadius: 2 }}
      />
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
            const hasBreakpoint = breakpointLines.has(lineNo);
            const isPaused = codePausedState?.line_number === lineNo;

            return (
              <div
                key={lineNo}
                className={`flex items-stretch ${
                  isPaused
                    ? "bg-red-900/40"
                    : isHighlighted
                      ? "bg-amber-900/30"
                      : "hover:bg-gray-900/50"
                }`}
              >
                {/* Breakpoint gutter */}
                <div
                  className="w-4 flex-shrink-0 flex items-center justify-center cursor-pointer select-none"
                  onClick={() => handleGutterClick(lineNo)}
                  title={hasBreakpoint ? "Remove breakpoint" : "Set breakpoint"}
                >
                  {hasBreakpoint ? (
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />
                  ) : (
                    <span className="w-2.5 h-2.5 rounded-full bg-transparent hover:bg-red-500/30 inline-block" />
                  )}
                </div>
                {/* Line number */}
                <span className="w-10 flex-shrink-0 text-right pr-2 text-gray-600 select-none">
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

        {/* Locals panel (collapsible, shown when paused) */}
        {animLocals && Object.keys(animLocals).length > 0 && (
          <div className="border-t border-gray-800">
            <button
              onClick={() => setShowLocals(!showLocals)}
              className="w-full px-3 py-1 text-[10px] text-gray-400 hover:text-gray-300 text-left flex items-center gap-1"
            >
              <span>{showLocals ? "\u25BC" : "\u25B6"}</span>
              <span>Locals</span>
            </button>
            {showLocals && (
              <div className="px-3 pb-2 max-h-32 overflow-auto">
                {Object.entries(animLocals).map(([key, val]) => (
                  <div key={key} className="flex gap-2 text-[10px] font-mono">
                    <span className="text-blue-400">{key}</span>
                    <span className="text-gray-600">=</span>
                    <span className="text-gray-400 truncate">
                      {typeof val === "object" ? JSON.stringify(val) : String(val)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}

export default memo(CodePanelNode);
