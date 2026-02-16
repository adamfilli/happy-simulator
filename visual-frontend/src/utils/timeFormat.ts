/** Shared time-unit formatting helpers. */

export interface TimeConfig {
  divisor: number;
  suffix: string;
  label: string;
}

const CONFIGS: Record<string, TimeConfig> = {
  s: { divisor: 1, suffix: "s", label: "Time (s)" },
  min: { divisor: 60, suffix: "min", label: "Time (min)" },
  h: { divisor: 3600, suffix: "h", label: "Time (h)" },
};

/** Return display config for the given unit (defaults to seconds). */
export function getTimeConfig(unit?: string): TimeConfig {
  return CONFIGS[unit ?? "s"] ?? CONFIGS.s;
}

/** Format a time value (always stored as seconds) into the display unit. */
export function formatTime(seconds: number, unit?: string, precision?: number): string {
  const cfg = getTimeConfig(unit);
  const value = seconds / cfg.divisor;
  const p = precision ?? (Math.abs(value) >= 100 ? 1 : 2);
  return value.toFixed(p) + cfg.suffix;
}
