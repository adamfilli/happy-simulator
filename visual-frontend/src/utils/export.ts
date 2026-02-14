/**
 * Export utility functions for the visual debugger.
 */

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function exportCanvasPng(canvas: HTMLCanvasElement, filename: string): void {
  canvas.toBlob((blob) => {
    if (blob) downloadBlob(blob, filename);
  }, "image/png");
}

export function exportCsv(times: number[], values: number[], filename: string): void {
  const lines = ["time_s,value"];
  for (let i = 0; i < times.length; i++) {
    lines.push(`${times[i]},${values[i]}`);
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  downloadBlob(blob, filename);
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
