# Visual Editor Design

This document describes the browser-based visual editor for creating simulation definitions.

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **React** + TypeScript | Best ecosystem for node editors |
| Node Editor | **React Flow** | Purpose-built, handles drag/drop/connect/zoom out of the box |
| State | **Zustand** | Simple, pairs well with React Flow |
| UI Components | **shadcn/ui** + Tailwind | Modern, customizable, fast to build |
| Build | **Vite** | Fast dev server, simple config |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser Editor                           │
├─────────────┬─────────────────────────────┬─────────────────────┤
│   Palette   │         Canvas              │   Properties        │
│             │                             │                     │
│  [Source]   │    ┌──────┐    ┌──────┐    │  Selected: Server   │
│  [Server]   │    │Source│───▶│Server│    │  ─────────────────  │
│  [Router]   │    └──────┘    └──────┘    │  service_time:      │
│  [Sink]     │        │           │       │  [exponential ▼]    │
│             │        ▼           ▼       │  mean: [0.05]       │
│  Drag to    │    ┌──────┐    ┌──────┐    │  concurrency: [4]   │
│  add...     │    │Router│───▶│ Sink │    │                     │
│             │    └──────┘    └──────┘    │  [Apply]            │
└─────────────┴─────────────────────────────┴─────────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   JSON Schema   │  ← Export / Save
                     │   (schema.md)   │
                     └─────────────────┘
```

---

## Key Components

### 1. Custom Node Types

Each entity type gets a custom React component:

```tsx
// nodes/QueuedServerNode.tsx
import { Handle, Position } from 'reactflow';

interface QueuedServerData {
  label: string;
  count?: number;
  params: {
    service_time: Distribution;
    concurrency: number;
  };
}

export function QueuedServerNode({ data, selected }: NodeProps<QueuedServerData>) {
  return (
    <div className={`node node-server ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} />

      <div className="node-header">
        <ServerIcon />
        <span>{data.label}</span>
        {data.count && data.count > 1 && (
          <span className="count-badge">×{data.count}</span>
        )}
      </div>

      <div className="node-body">
        <div className="stat">μ = {data.params.service_time.mean}s</div>
        <div className="stat">c = {data.params.concurrency}</div>
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  );
}
```

### 2. Node Registry

Maps our schema types to React components:

```tsx
// nodes/registry.ts
import { QueuedServerNode } from './QueuedServerNode';
import { SourceNode } from './SourceNode';
import { RouterNode } from './RouterNode';
import { SinkNode } from './SinkNode';

export const nodeTypes = {
  QueuedServer: QueuedServerNode,
  Source: SourceNode,
  RandomRouter: RouterNode,
  Sink: SinkNode,
  // ... add more as needed
};

// Default params for each type (used when dragging from palette)
export const nodeDefaults: Record<string, object> = {
  QueuedServer: {
    service_time: { type: 'exponential', mean: 0.05 },
    concurrency: 1,
  },
  Source: {
    arrival: { type: 'poisson', rate: 10 },
    duration: 60,
  },
  // ...
};
```

### 3. Main Editor Component

```tsx
// Editor.tsx
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
} from 'reactflow';
import { nodeTypes } from './nodes/registry';
import { Palette } from './Palette';
import { PropertiesPanel } from './PropertiesPanel';
import { serializeToSchema, deserializeFromSchema } from './serialization';

export function Editor() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Handle new connections
  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge(params, eds));
  }, []);

  // Handle node selection
  const onNodeClick = useCallback((_, node) => {
    setSelectedNode(node);
  }, []);

  // Drag from palette to canvas
  const onDrop = useCallback((event) => {
    const type = event.dataTransfer.getData('application/reactflow');
    const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });

    const newNode = {
      id: `${type}-${Date.now()}`,
      type,
      position,
      data: {
        label: type,
        params: nodeDefaults[type]
      },
    };

    setNodes((nds) => [...nds, newNode]);
  }, []);

  // Export to our JSON schema
  const handleExport = () => {
    const schema = serializeToSchema(nodes, edges);
    console.log(JSON.stringify(schema, null, 2));
    // or download as file, send to API, etc.
  };

  // Load from JSON schema
  const handleImport = (json: string) => {
    const schema = JSON.parse(json);
    const { nodes, edges } = deserializeFromSchema(schema);
    setNodes(nodes);
    setEdges(edges);
  };

  return (
    <div className="editor-container">
      <Palette />

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>

      <PropertiesPanel
        node={selectedNode}
        onChange={(data) => updateNodeData(selectedNode.id, data)}
      />
    </div>
  );
}
```

### 4. Serialization (React Flow ↔ JSON Schema)

```tsx
// serialization.ts
import { Node, Edge } from 'reactflow';
import { SimulationSchema } from './types';

export function serializeToSchema(nodes: Node[], edges: Edge[]): SimulationSchema {
  const entities = [];
  const sources = [];
  const connections = [];

  for (const node of nodes) {
    const base = {
      id: node.id,
      label: node.data.label,
      position: { x: node.position.x, y: node.position.y },
    };

    if (node.type === 'Source') {
      sources.push({
        ...base,
        target: findTargetFromEdges(node.id, edges),
        arrival: node.data.params.arrival,
        profile: node.data.params.profile,
        duration: node.data.params.duration,
      });
    } else {
      entities.push({
        ...base,
        type: node.type,
        count: node.data.count,
        params: node.data.params,
      });
    }
  }

  // Convert edges to connections
  for (const edge of edges) {
    // Skip source->entity connections (handled in source.target)
    const sourceNode = nodes.find(n => n.id === edge.source);
    if (sourceNode?.type === 'Source') continue;

    connections.push({
      from: edge.source,
      to: edge.target,
      routing: edge.data?.routing,
    });
  }

  return {
    simulation: { name: 'Untitled Simulation' },
    entities,
    sources,
    connections,
    probes: [], // TODO: add probe UI
  };
}

export function deserializeFromSchema(schema: SimulationSchema): { nodes: Node[], edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Create entity nodes
  for (const entity of schema.entities || []) {
    nodes.push({
      id: entity.id,
      type: entity.type,
      position: entity.position || { x: 0, y: 0 },
      data: {
        label: entity.label || entity.id,
        count: entity.count,
        params: entity.params,
      },
    });
  }

  // Create source nodes + edges to their targets
  for (const source of schema.sources || []) {
    nodes.push({
      id: source.id,
      type: 'Source',
      position: source.position || { x: 0, y: 0 },
      data: {
        label: source.label || source.id,
        params: {
          arrival: source.arrival,
          profile: source.profile,
          duration: source.duration,
        },
      },
    });

    // Source -> target edge
    if (source.target) {
      edges.push({
        id: `${source.id}->${source.target}`,
        source: source.id,
        target: source.target,
      });
    }
  }

  // Create connection edges
  for (const conn of schema.connections || []) {
    const targets = Array.isArray(conn.to) ? conn.to : [conn.to];
    for (const target of targets) {
      edges.push({
        id: `${conn.from}->${target}`,
        source: conn.from,
        target: target,
        data: { routing: conn.routing },
      });
    }
  }

  return { nodes, edges };
}
```

### 5. Properties Panel

```tsx
// PropertiesPanel.tsx
export function PropertiesPanel({ node, onChange }: Props) {
  if (!node) {
    return <div className="properties-panel empty">Select a node</div>;
  }

  return (
    <div className="properties-panel">
      <h3>{node.data.label}</h3>
      <p className="type-label">{node.type}</p>

      <div className="form">
        <label>Label</label>
        <input
          value={node.data.label}
          onChange={(e) => onChange({ ...node.data, label: e.target.value })}
        />

        {node.type === 'QueuedServer' && (
          <>
            <label>Count (fleet size)</label>
            <input
              type="number"
              value={node.data.count || 1}
              onChange={(e) => onChange({ ...node.data, count: parseInt(e.target.value) })}
            />

            <label>Service Time</label>
            <DistributionEditor
              value={node.data.params.service_time}
              onChange={(v) => onChange({
                ...node.data,
                params: { ...node.data.params, service_time: v }
              })}
            />

            <label>Concurrency</label>
            <input
              type="number"
              value={node.data.params.concurrency}
              onChange={(e) => onChange({
                ...node.data,
                params: { ...node.data.params, concurrency: parseInt(e.target.value) }
              })}
            />
          </>
        )}

        {node.type === 'Source' && (
          <SourceParamsEditor params={node.data.params} onChange={...} />
        )}

        {/* etc for other types */}
      </div>
    </div>
  );
}
```

---

## Project Structure

```
visual/
├── editor/                    # React app
│   ├── src/
│   │   ├── components/
│   │   │   ├── Editor.tsx          # Main canvas
│   │   │   ├── Palette.tsx         # Draggable node types
│   │   │   ├── PropertiesPanel.tsx # Edit selected node
│   │   │   ├── Toolbar.tsx         # Save/Load/Run buttons
│   │   │   └── DistributionEditor.tsx
│   │   ├── nodes/
│   │   │   ├── registry.ts         # Node type mapping
│   │   │   ├── SourceNode.tsx
│   │   │   ├── QueuedServerNode.tsx
│   │   │   ├── RouterNode.tsx
│   │   │   └── SinkNode.tsx
│   │   ├── serialization/
│   │   │   ├── serialize.ts        # React Flow → JSON schema
│   │   │   └── deserialize.ts      # JSON schema → React Flow
│   │   ├── types/
│   │   │   └── schema.ts           # TypeScript types matching schema.md
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
├── schema.md                  # JSON schema definition
├── Render.md                  # Runtime architecture
└── Editor.md                  # This file
```

---

## Getting Started

```bash
# Create the React app
cd visual
npm create vite@latest editor -- --template react-ts
cd editor

# Install dependencies
npm install reactflow zustand
npm install -D tailwindcss postcss autoprefixer
npm install @radix-ui/react-select @radix-ui/react-popover  # for UI

# Initialize Tailwind
npx tailwindcss init -p
```

---

## Integration with Backend

```tsx
// api.ts
const API_URL = 'http://localhost:8000';

export async function runSimulation(schema: SimulationSchema) {
  const response = await fetch(`${API_URL}/simulations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(schema),
  });
  return response.json();  // { id, state }
}

export function streamSimulation(simId: string, onUpdate: (state) => void) {
  const ws = new WebSocket(`ws://localhost:8000/simulations/${simId}/stream`);
  ws.onmessage = (event) => {
    onUpdate(JSON.parse(event.data));
  };
  return ws;
}
```

---

## Key Features to Build

| Priority | Feature | Effort |
|----------|---------|--------|
| 1 | Drag nodes from palette | Low (React Flow built-in) |
| 1 | Connect nodes with edges | Low (React Flow built-in) |
| 1 | Properties panel for selected node | Medium |
| 1 | Export to JSON | Low |
| 1 | Import from JSON | Low |
| 2 | Custom node styling per type | Medium |
| 2 | Fleet count badge on nodes | Low |
| 2 | Validate connections (type checking) | Medium |
| 2 | Undo/redo | Medium (Zustand middleware) |
| 3 | Run simulation button | Low (API call) |
| 3 | Live metrics overlay on nodes | Medium |
| 3 | Probe configuration UI | Medium |

---

## Node Styling

Each node type should have distinct visual treatment:

| Type | Color | Icon | Shape |
|------|-------|------|-------|
| Source | Green | Play ▶ | Rounded left edge |
| QueuedServer | Blue | Server 🖥 | Rectangle |
| RandomRouter | Orange | Split ⑂ | Diamond or rectangle |
| Sink | Red | Stop ■ | Rounded right edge |
| Delay | Gray | Clock ⏱ | Rectangle |
| Custom | Purple | Code { } | Dashed border |

---

## Live Simulation Overlay

When simulation is running, overlay metrics on nodes:

```
┌─────────────────────┐
│  Server             │
│  ×1000              │
├─────────────────────┤
│  depth: 847         │  ← Live metric
│  processed: 12,453  │  ← Live metric
│  ▓▓▓▓▓▓▓▓░░ 84%    │  ← Utilization bar
└─────────────────────┘
```

This requires:
1. WebSocket connection to backend
2. State updates mapped to node IDs
3. Re-render nodes with live data
