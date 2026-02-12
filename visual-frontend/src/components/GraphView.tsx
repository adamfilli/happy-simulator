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
import EntityNode from "./EntityNode";

const elk = new ELK();

const CATEGORY_COLORS: Record<string, string> = {
  source: "#22c55e",
  queued_resource: "#3b82f6",
  sink: "#ef4444",
  rate_limiter: "#f97316",
  router: "#a855f7",
  resource: "#a16207",
  other: "#6b7280",
};

const nodeTypes = { entity: EntityNode };

export default function GraphView() {
  const topology = useSimStore((s) => s.topology);
  const state = useSimStore((s) => s.state);
  const selectEntity = useSimStore((s) => s.selectEntity);
  const selectedEntity = useSimStore((s) => s.selectedEntity);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [layoutDone, setLayoutDone] = useState(false);

  // Build edges from topology
  const flowEdges: Edge[] = useMemo(() => {
    if (!topology) return [];
    return topology.edges.map((e, i) => ({
      id: `e-${e.source}-${e.target}-${i}`,
      source: e.source,
      target: e.target,
      animated: false,
      style: { stroke: "#4b5563", strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#4b5563", width: 16, height: 16 },
    }));
  }, [topology]);

  // Run elkjs layout when topology changes
  useEffect(() => {
    if (!topology || topology.nodes.length === 0) return;

    const graph = {
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.spacing.nodeNode": "60",
        "elk.layered.spacing.nodeNodeBetweenLayers": "100",
      },
      children: topology.nodes.map((n) => ({
        id: n.id,
        width: 180,
        height: 80,
      })),
      edges: topology.edges.map((e, i) => ({
        id: `elk-${i}`,
        sources: [e.source],
        targets: [e.target],
      })),
    };

    elk.layout(graph).then((laid) => {
      const rfNodes: Node[] = (laid.children ?? []).map((child) => {
        const topoNode = topology.nodes.find((n) => n.id === child.id)!;
        return {
          id: child.id,
          type: "entity",
          position: { x: child.x ?? 0, y: child.y ?? 0 },
          data: {
            label: child.id,
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

  // Update edges when topology edges change (dynamic discovery)
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

  const onNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
      selectEntity(node.id);
    },
    [selectEntity]
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      nodeTypes={nodeTypes}
      fitView
      minZoom={0.3}
      maxZoom={2}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#1f2937" gap={20} size={1} />
    </ReactFlow>
  );
}
