# react-grid-layout Upgrade

## Summary

Replaced the custom drag implementation in the Dashboard view with [react-grid-layout](https://github.com/react-grid-layout/react-grid-layout) v2 for snap-to-grid positioning, resize handles, and automatic vertical compaction.

## What Changed

### Dependencies

- Added `react-grid-layout` v2.2.2 (includes built-in TypeScript types)
- Removed `@types/react-grid-layout` (v1.x types, incompatible with v2)

### `types.ts` — `DashboardPanelConfig`

Added `w` (width) and `h` (height) grid unit fields alongside existing `x`/`y`. All four are now in react-grid-layout grid units (not pixels).

### `useSimState.ts` — Store

- Removed `moveDashboardPanel(id, x, y)` (custom pixel-based drag)
- Added `updateDashboardLayout(layout)` — takes the full layout array from react-grid-layout's `onLayoutChange` and syncs `x/y/w/h` back to each panel

### `DashboardView.tsx` — Grid Container

- Uses `GridLayout` component with `useContainerWidth()` hook for reactive width
- Grid config: 12 columns, 50px row height, 12px margins
- Drag restricted to `.drag-handle` CSS class via `dragConfig`
- Vertical compaction enabled
- Panels mapped to `Layout` items with `minW: 2`, `minH: 2`

### `DashboardPanel.tsx` — Grid Item

- Removed all custom drag logic (dragRef, mousedown/mousemove/mouseup listeners)
- Removed absolute positioning props (`x`, `y`, fixed `width: 400`)
- Panel fills its grid cell (`h-full`) with flex column layout
- Title bar has `drag-handle` CSS class for react-grid-layout to hook into
- Accepts `onClose` callback prop instead of calling store directly

### `InspectorPanel.tsx` — Pin Button

- Updated `addDashboardPanel` call to include `w: 4, h: 4` grid units
- Placement uses `col % 3 * 4` / `floor(count / 3) * 4` for row-fill pattern

## v2 API Notes

react-grid-layout v2 removed the `WidthProvider` HOC. Use the `useContainerWidth()` hook instead:

```tsx
const { width, containerRef, mounted } = useContainerWidth();
// ...
<div ref={containerRef}>
  {mounted && <GridLayout width={width} ... />}
</div>
```

`GridLayout` uses grouped config objects:
- `gridConfig: { cols, rowHeight, margin, containerPadding, maxRows }`
- `dragConfig: { enabled, handle, cancel, threshold }`
- `resizeConfig: { enabled, handles, handleComponent }`

`Layout` is `readonly LayoutItem[]`. `onLayoutChange` receives `(layout: Layout)`.
