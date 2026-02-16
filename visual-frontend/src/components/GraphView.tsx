import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  type Node,
  type Edge,
  type NodeMouseHandler,
  useNodesState,
  useEdgesState,
  useReactFlow,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";
import { useSimStore } from "../hooks/useSimState";
import { useWebSocket } from "../hooks/useWebSocket";
import EntityNode from "./EntityNode";
import GroupNode from "./GroupNode";
import AnimatedEdge from "./AnimatedEdge";
import ChartNode from "./ChartNode";
import CodePanelNode from "./CodePanelNode";
import { CodePanelCtx, type CodePanelContext } from "./CodePanelContext";
import type { PinnedChart } from "../types";

const elk = new ELK();

const CATEGORY_COLORS: Record<string, string> = {
  source: "#22c55e",
  queued_resource: "#3b82f6",
  sink: "#ef4444",
  rate_limiter: "#f97316",
  router: "#a855f7",
  resource: "#a16207",
  probe: "#06b6d4",
  other: "#6b7280",
};

const nodeTypes = { entity: EntityNode, group: GroupNode, chart: ChartNode, codePanel: CodePanelNode };
const edgeTypes = { animated: AnimatedEdge };

const DRAG_MIME = "application/happysim-chart";

export default function GraphView() {
  return (
    <ReactFlowProvider>
      <GraphViewInner />
    </ReactFlowProvider>
  );
}

function pinnedChartToNode(chart: PinnedChart): Node {
  return {
    id: `chart-${chart.id}`,
    type: "chart",
    position: { x: chart.x, y: chart.y },
    data: {
      chartId: chart.id,
      kind: chart.kind,
      entityName: chart.entityName,
      metricKey: chart.metricKey,
      displayMode: chart.displayMode,
      probeName: chart.probeName,
      label: chart.label,
    },
  };
}

function GraphViewInner() {
  const topology = useSimStore((s) => s.topology);
  const state = useSimStore((s) => s.state);
  const selectEntity = useSimStore((s) => s.selectEntity);
  const selectedEntity = useSimStore((s) => s.selectedEntity);
  const codePanels = useSimStore((s) => s.codePanels);
  const pinnedCharts = useSimStore((s) => s.pinnedCharts);
  const addPinnedChart = useSimStore((s) => s.addPinnedChart);
  const updatePinnedChartPosition = useSimStore((s) => s.updatePinnedChartPosition);
  const extractedEntities = useSimStore((s) => s.extractedEntities);
  const { send } = useWebSocket();
  const { screenToFlowPosition } = useReactFlow();
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [layoutDone, setLayoutDone] = useState(false);

  const handleOpenCodePanel = useCallback(
    (entityName: string) => {
      if (codePanels.has(entityName)) {
        // Already open — close it
        send("deactivate_code_debug", { entity_name: entityName });
      } else {
        send("activate_code_debug", { entity_name: entityName });
      }
    },
    [codePanels, send]
  );

  const handleCloseCodePanel = useCallback(
    (entityName: string) => {
      send("deactivate_code_debug", { entity_name: entityName });
    },
    [send]
  );

  // Build edges from topology
  const flowEdges: Edge[] = useMemo(() => {
    if (!topology) return [];
    return topology.edges.map((e, i) => {
      const isProbe = e.kind === "probe";
      return {
        id: `e-${e.source}-${e.target}-${i}`,
        source: e.source,
        target: e.target,
        sourceHandle: isProbe ? "bottom" : "right",
        targetHandle: isProbe ? "top" : "left",
        type: "animated",
        data: {
          isProbe,
          statsKey: `${e.source}->${e.target}`,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: isProbe ? "#06b6d4" : "#4b5563",
          width: 16,
          height: 16,
        },
      };
    });
  }, [topology]);

  // Run elkjs layout when topology changes
  useEffect(() => {
    if (!topology || topology.nodes.length === 0) return;

    // Separate probe nodes/edges from the main data-flow graph
    const probeNodeIds = new Set(
      topology.nodes.filter((n) => n.category === "probe").map((n) => n.id)
    );
    const dataNodes = topology.nodes.filter((n) => !probeNodeIds.has(n.id));
    const dataEdges = topology.edges.filter((e) => e.kind !== "probe" && !probeNodeIds.has(e.source));

    const graph = {
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.spacing.nodeNode": "60",
        "elk.layered.spacing.nodeNodeBetweenLayers": "100",
      },
      children: dataNodes.map((n) => ({
        id: n.id,
        width: n.is_group ? 220 : 180,
        height: n.is_group ? 100 : 80,
      })),
      edges: dataEdges.map((e, i) => ({
        id: `elk-${i}`,
        sources: [e.source],
        targets: [e.target],
      })),
    };

    elk.layout(graph).then((laid) => {
      // Build position map from laid-out data nodes
      const posMap = new Map<string, { x: number; y: number }>();
      for (const child of laid.children ?? []) {
        posMap.set(child.id, { x: child.x ?? 0, y: child.y ?? 0 });
      }

      // Position probes above their target so the bottom→top arrow flows downward
      const PROBE_Y_OFFSET = 130;
      for (const probeId of probeNodeIds) {
        const probeEdge = topology.edges.find((e) => e.source === probeId && e.kind === "probe");
        if (probeEdge) {
          const targetPos = posMap.get(probeEdge.target);
          if (targetPos) {
            posMap.set(probeId, { x: targetPos.x, y: targetPos.y - PROBE_Y_OFFSET });
          } else {
            posMap.set(probeId, { x: 0, y: -PROBE_Y_OFFSET });
          }
        } else {
          posMap.set(probeId, { x: 0, y: -PROBE_Y_OFFSET });
        }
      }

      const rfNodes: Node[] = topology.nodes.map((topoNode) => {
        const pos = posMap.get(topoNode.id) ?? { x: 0, y: 0 };
        if (topoNode.is_group) {
          return {
            id: topoNode.id,
            type: "group",
            position: { x: pos.x, y: pos.y },
            style: { background: "none", border: "none", padding: 0 },
            data: {
              label: topoNode.type,
              entityType: topoNode.type,
              category: topoNode.category,
              color: CATEGORY_COLORS[topoNode.category] || CATEGORY_COLORS.other,
              memberCount: topoNode.member_count ?? 0,
            },
          };
        }
        return {
          id: topoNode.id,
          type: "entity",
          position: { x: pos.x, y: pos.y },
          data: {
            label: topoNode.id,
            entityType: topoNode.type,
            category: topoNode.category,
            color: CATEGORY_COLORS[topoNode.category] || CATEGORY_COLORS.other,
            metrics: {},
          },
        };
      });
      // Preserve existing pinned chart nodes across ELK re-layout
      const chartNodes = useSimStore.getState().pinnedCharts.map(pinnedChartToNode);
      setNodes([...rfNodes, ...chartNodes]);
      setEdges(flowEdges);
      setLayoutDone(true);
    });
  }, [topology, flowEdges, setNodes, setEdges]);

  // Update edges when topology edges or edge stats change (dynamic discovery)
  useEffect(() => {
    if (layoutDone) {
      setEdges(flowEdges);
    }
  }, [flowEdges, layoutDone, setEdges]);

  // Update node metrics when state changes
  useEffect(() => {
    if (!state) return;
    setNodes((nds) =>
      nds.map((n) => {
        if (n.type === "codePanel" || n.type === "chart") return n;
        if (n.type === "group") {
          return {
            ...n,
            data: { ...n.data, selected: n.id === selectedEntity },
          };
        }
        const entityState = state.entities[n.id];
        return {
          ...n,
          data: {
            ...n.data,
            metrics: entityState || {},
            selected: n.id === selectedEntity,
          },
        };
      })
    );
  }, [state, selectedEntity, setNodes]);

  // Manage code panel nodes and edges
  useEffect(() => {
    setNodes((nds) => {
      // Remove closed panels
      const withoutStale = nds.filter(
        (n) => n.type !== "codePanel" || codePanels.has(n.data.entityName as string)
      );

      // Add new panels
      const existingPanelIds = new Set(
        withoutStale.filter((n) => n.type === "codePanel").map((n) => n.id)
      );

      const newPanels: Node[] = [];
      for (const [entityName, config] of codePanels) {
        const panelId = `code-${entityName}`;
        if (!existingPanelIds.has(panelId)) {
          // Position the code panel to the right of the entity node
          const entityNode = withoutStale.find((n) => n.id === entityName);
          const x = entityNode ? entityNode.position.x + 220 : 400;
          const y = entityNode ? entityNode.position.y : 0;

          newPanels.push({
            id: panelId,
            type: "codePanel",
            position: { x, y },
            data: {
              entityName,
              classname: config.source.class_name,
              methodName: config.source.method_name,
              sourceLines: config.source.source_lines,
              startLine: config.source.start_line,
              onClose: handleCloseCodePanel,
            },
            style: { width: 450, height: 350 },
            dragHandle: ".drag-handle",
          });
        }
      }

      return [...withoutStale, ...newPanels];
    });

    // Update edges for code panels
    setEdges((eds) => {
      const withoutCodeEdges = eds.filter((e) => !e.id.startsWith("code-edge-"));
      const codeEdges: Edge[] = [];
      for (const entityName of codePanels.keys()) {
        codeEdges.push({
          id: `code-edge-${entityName}`,
          source: entityName,
          target: `code-${entityName}`,
          sourceHandle: "right",
          targetHandle: "code-left",
          style: { stroke: "#d97706", strokeWidth: 1.5, strokeDasharray: "4 3" },
          animated: true,
        });
      }
      return [...withoutCodeEdges, ...codeEdges];
    });
  }, [codePanels, handleCloseCodePanel, setNodes, setEdges]);

  // Sync pinned charts → React Flow nodes
  useEffect(() => {
    setNodes((nds) => {
      const pinnedIds = new Set(pinnedCharts.map((c) => `chart-${c.id}`));
      // Remove stale chart nodes
      const withoutStale = nds.filter((n) => n.type !== "chart" || pinnedIds.has(n.id));
      // Add new chart nodes
      const existingChartIds = new Set(
        withoutStale.filter((n) => n.type === "chart").map((n) => n.id)
      );
      const newChartNodes = pinnedCharts
        .filter((c) => !existingChartIds.has(`chart-${c.id}`))
        .map(pinnedChartToNode);
      // Update displayMode on existing chart nodes
      const updated = withoutStale.map((n) => {
        if (n.type !== "chart") return n;
        const chartId = (n.data as { chartId: string }).chartId;
        const chart = pinnedCharts.find((c) => c.id === chartId);
        if (chart && chart.displayMode !== (n.data as { displayMode?: string }).displayMode) {
          return { ...n, data: { ...n.data, displayMode: chart.displayMode } };
        }
        return n;
      });
      return [...updated, ...newChartNodes];
    });
  }, [pinnedCharts, setNodes]);

  // Sync extracted entities → React Flow nodes and dashed "member-of" edges
  useEffect(() => {
    setNodes((nds) => {
      const extractedIds = new Set([...extractedEntities.keys()].map((n) => `extracted-${n}`));
      // Remove retracted entities
      const withoutStale = nds.filter(
        (n) => !n.id.startsWith("extracted-") || extractedIds.has(n.id)
      );
      // Add new extracted entities
      const existingIds = new Set(withoutStale.map((n) => n.id));
      const newNodes: Node[] = [];
      for (const [entityName, groupId] of extractedEntities) {
        const nodeId = `extracted-${entityName}`;
        if (!existingIds.has(nodeId)) {
          // Position near the group node
          const groupNode = withoutStale.find((n) => n.id === groupId);
          const x = groupNode ? groupNode.position.x + 250 : 400;
          const y = groupNode
            ? groupNode.position.y + (newNodes.length * 100)
            : newNodes.length * 100;
          newNodes.push({
            id: nodeId,
            type: "entity",
            position: { x, y },
            data: {
              label: entityName,
              entityType: "",
              category: groupNode?.data?.category ?? "other",
              color: (groupNode?.data?.color as string) ?? CATEGORY_COLORS.other,
              metrics: {},
            },
          });
        }
      }
      return [...withoutStale, ...newNodes];
    });

    // Update edges: add dashed "member-of" edges
    setEdges((eds) => {
      const withoutExtracted = eds.filter((e) => !e.id.startsWith("extracted-edge-"));
      const extractedEdges: Edge[] = [];
      for (const [entityName, groupId] of extractedEntities) {
        extractedEdges.push({
          id: `extracted-edge-${entityName}`,
          source: groupId,
          target: `extracted-${entityName}`,
          sourceHandle: "right",
          targetHandle: "left",
          style: { stroke: "#6b7280", strokeWidth: 1.5, strokeDasharray: "6 4" },
        });
      }
      return [...withoutExtracted, ...extractedEdges];
    });
  }, [extractedEntities, setNodes, setEdges]);

  // Fetch state for extracted entities on each tick — batched into one setNodes call
  useEffect(() => {
    if (!state || extractedEntities.size === 0) return;
    const names = [...extractedEntities.keys()];
    Promise.all(
      names.map((name) =>
        fetch(`/api/entity_state?entity=${encodeURIComponent(name)}`)
          .then((r) => r.json())
          .then((d: { entity: string; state: Record<string, unknown> }) => d)
      )
    ).then((results) => {
      const stateMap = new Map(results.map((r) => [`extracted-${r.entity}`, r.state]));
      setNodes((nds) =>
        nds.map((n) => {
          const s = stateMap.get(n.id);
          return s ? { ...n, data: { ...n.data, metrics: s } } : n;
        })
      );
    });
  }, [state?.events_processed, extractedEntities, setNodes, state]);

  // Drop handler for inspector drag
  const onDragOver = useCallback((e: React.DragEvent) => {
    if (e.dataTransfer.types.includes(DRAG_MIME)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      const raw = e.dataTransfer.getData(DRAG_MIME);
      if (!raw) return;
      e.preventDefault();
      const payload = JSON.parse(raw) as Omit<PinnedChart, "id" | "x" | "y">;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      addPinnedChart({
        ...payload,
        id: crypto.randomUUID(),
        x: position.x,
        y: position.y,
      });
    },
    [screenToFlowPosition, addPinnedChart]
  );

  // Persist chart node position after drag
  const onNodeDragStop = useCallback(
    (_: unknown, node: Node) => {
      if (node.type === "chart") {
        const chartId = (node.data as { chartId: string }).chartId;
        updatePinnedChartPosition(chartId, node.position.x, node.position.y);
      }
    },
    [updatePinnedChartPosition]
  );

  // Click handler — skip chart nodes, handle extracted entities
  const onNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
      if (node.type === "chart") return;
      // For extracted entities, select with the original entity name
      if (node.id.startsWith("extracted-")) {
        selectEntity(node.id.replace("extracted-", ""));
        return;
      }
      selectEntity(node.id);
    },
    [selectEntity]
  );

  const codePanelCtx = useMemo<CodePanelContext>(
    () => ({ onOpenCodePanel: handleOpenCodePanel, openPanels: new Set(codePanels.keys()) }),
    [handleOpenCodePanel, codePanels]
  );

  return (
    <CodePanelCtx.Provider value={codePanelCtx}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onNodeDragStop={onNodeDragStop}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        minZoom={0.3}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1f2937" gap={20} size={1} />
      </ReactFlow>
    </CodePanelCtx.Provider>
  );
}
