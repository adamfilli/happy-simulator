export type Series = { times: number[]; values: number[] };

/**
 * Convert raw cumulative values to per-second rate.
 * values[i] = (v[i+1] - v[i]) / (t[i+1] - t[i]), times at midpoints.
 */
export function toRate(s: Series): Series {
  const n = s.times.length;
  if (n < 2) return { times: [], values: [] };

  const times: number[] = [];
  const values: number[] = [];

  for (let i = 0; i < n - 1; i++) {
    const dt = s.times[i + 1] - s.times[i];
    if (dt <= 0) continue;
    times.push((s.times[i] + s.times[i + 1]) / 2);
    values.push((s.values[i + 1] - s.values[i]) / dt);
  }

  return { times, values };
}

/**
 * Bucket samples into windowS-wide time buckets and apply an aggregate.
 */
export function toBucketed(
  s: Series,
  windowS: number,
  fn: "avg" | "p99",
): Series {
  const n = s.times.length;
  if (n === 0) return { times: [], values: [] };

  const tMin = s.times[0];
  const tMax = s.times[n - 1];

  const times: number[] = [];
  const values: number[] = [];

  let bucketStart = tMin;
  let idx = 0;

  while (bucketStart < tMax) {
    const bucketEnd = bucketStart + windowS;
    const bucketVals: number[] = [];

    while (idx < n && s.times[idx] < bucketEnd) {
      bucketVals.push(s.values[idx]);
      idx++;
    }

    if (bucketVals.length > 0) {
      times.push(bucketStart + windowS / 2);
      values.push(fn === "avg" ? avg(bucketVals) : p99(bucketVals));
    }

    bucketStart = bucketEnd;
  }

  return { times, values };
}

function avg(vals: number[]): number {
  let sum = 0;
  for (const v of vals) sum += v;
  return sum / vals.length;
}

function p99(vals: number[]): number {
  const sorted = vals.slice().sort((a, b) => a - b);
  const idx = Math.ceil(sorted.length * 0.99) - 1;
  return sorted[Math.max(0, idx)];
}
