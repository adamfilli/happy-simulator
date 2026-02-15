# Plan: Drag-and-Drop Pinned Chart Tiles on Graph View

## Context

The visual debugger's Inspector panel shows sparkline charts (per-metric) and probe time series when an entity is selected. Currently these are view-only within the inspector. The user wants to **drag charts from the inspector and drop them onto the Graph canvas**, where they become freely-positionable, live-updating tiles that sit alongside entity nodes and pan/zoom with the graph.

## Approach

Use **React Flow custom nodes** for chart tiles (they inherit pan/zoom/drag for free) and **HTML5 native drag API** for the inspector-to-graph drop interaction. No new npm dependencies needed.

## Files Changed

| Order | File | Action |
|-------|------|--------|
| 1 | `visual-frontend/src/types.ts` | Add `PinnedChart` interface |
| 2 | `visual-frontend/src/hooks/useSimState.ts` | Add `pinnedCharts` state + 4 actions, update `reset` |
| 3 | `visual-frontend/src/components/ChartNode.tsx` | **New** - React Flow custom node rendering a live chart |
| 4 | `visual-frontend/src/components/GraphView.tsx` | Register chart node type, drop handler, ELK survival, drag-stop sync |
| 5 | `visual-frontend/src/components/InspectorPanel.tsx` | Add `draggable` + `onDragStart` on sparklines and probe charts |

No backend changes needed - existing `/api/entity_history` and `/api/timeseries` endpoints already serve the data.

## Step 1: Types (`types.ts`)

Add after `DashboardPanelConfig`:

```typescript
export interface PinnedChart {
  id: string;
  x: number;              // React Flow canvas coordinates
  y: number;
  kind: "entity_metric" | "probe";
  entityName?: string;     // for entity metrics
  metricKey?: string;      // for entity metrics
  displayMode?: "total" | "rate" | "avg" | "p99";
  probeName?: string;      // for probes
  label: string;
}
```

## Step 2: Store (`useSimState.ts`)

- Add `pinnedCharts: PinnedChart[]` to state
- Add actions: `addPinnedChart`, `removePinnedChart`, `updatePinnedChartPosition`, `updatePinnedChartMode`
- Include `pinnedCharts: []` in `reset()` action

## Step 3: ChartNode Component (new file `ChartNode.tsx`)

- 280x170px fixed-size React Flow custom node
- Title bar (28px) - label + mode toggle (entity metrics only) + close button
- Chart area (140px) - `TimeSeriesChart` component with `nodrag` CSS class (prevents React Flow drag, allows chart hover/zoom)
- Fetches from `/api/entity_history` or `/api/timeseries` keyed on `events_processed` (same pattern as `DashboardPanel`)
- Applies sparkline transforms (total/rate/avg/p99) using existing `toRate`/`toBucketed` from `sparklineTransforms.ts`
- Color: green for entity metrics, blue for probes

## Step 4: GraphView Changes (`GraphView.tsx`)

Key changes:
1. **Register node type**: `{ entity: EntityNode, chart: ChartNode }`
2. **Drop handler**: `onDragOver` + `onDrop` on `<ReactFlow>`, using `screenToFlowPosition()` from `useReactFlow()` to place chart at drop position
3. **ELK survival**: After ELK computes entity positions, append chart nodes from `useSimStore.getState().pinnedCharts` (imperative read to avoid re-triggering ELK)
4. **Sync effect**: Separate `useEffect` on `pinnedCharts` merges chart nodes with entity nodes via `setNodes(prev => ...)`
5. **Metrics update guard**: Add `if (n.type === "chart") return n;` to skip chart nodes in the entity metrics update effect
6. **Drag-stop sync**: `onNodeDragStop` persists chart node positions back to store
7. **Click guard**: Prevent chart node clicks from triggering entity selection

Chart node IDs use `chart-{id}` prefix to avoid collision with entity names.

## Step 5: InspectorPanel Changes (`InspectorPanel.tsx`)

- **Sparkline rows**: Add `draggable={hasSparkline}` + `onDragStart` that serializes `{kind, entityName, metricKey, displayMode, label}` to `application/happysim-chart` MIME type. Add `cursor-grab` styling.
- **Probe chart section**: Wrap in draggable container with same `onDragStart` pattern, serializing `{kind: "probe", probeName, label}`.
- Add `title="Drag to graph to pin"` for discoverability.

## Verification

1. Run `cd visual-frontend && npm run build` to verify TypeScript compilation
2. Launch debugger: `python examples/visual_debugger.py`
3. Test: click entity with metrics, drag a sparkline onto graph -> chart tile appears at drop position
4. Test: click a probe, drag probe chart onto graph -> chart tile appears
5. Test: chart tiles update live during simulation play
6. Test: drag chart tiles to reposition, verify they stay put
7. Test: click mode toggle on entity metric chart tile, verify transform changes
8. Test: close button removes chart tile
9. Test: reset simulation -> chart tiles are cleared
10. Test: chart tiles pan/zoom with graph
