import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  type Node,
  type Edge,
  type NodeMouseHandler,
  useNodesState,
  useEdgesState,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";
import { useSimStore } from "../hooks/useSimState";
import { useWebSocket } from "../hooks/useWebSocket";
import EntityNode from "./EntityNode";
import AnimatedEdge from "./AnimatedEdge";
import CodePanelNode from "./CodePanelNode";
import { CodePanelCtx, type CodePanelContext } from "./CodePanelContext";

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

const nodeTypes = { entity: EntityNode, codePanel: CodePanelNode };
const edgeTypes = { animated: AnimatedEdge };

export default function GraphView() {
  const topology = useSimStore((s) => s.topology);
  const state = useSimStore((s) => s.state);
  const selectEntity = useSimStore((s) => s.selectEntity);
  const selectedEntity = useSimStore((s) => s.selectedEntity);
  const codePanels = useSimStore((s) => s.codePanels);
  const { send } = useWebSocket();
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
        width: 180,
        height: 80,
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
      setNodes(rfNodes);
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
        if (n.type === "codePanel") return n;
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

  const onNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
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
