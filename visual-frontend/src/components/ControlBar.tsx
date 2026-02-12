import { useState } from "react";
import { useSimStore } from "../hooks/useSimState";

const SPEEDS = [1, 5, 10, 50, 100];

interface Props {
  onStep: (count: number) => void;
  onPlay: (speed: number) => void;
  onPause: () => void;
  onReset: () => void;
  onRunTo: (time_s: number) => void;
  onRunToEvent: (n: number) => void;
}

export default function ControlBar({ onStep, onPlay, onPause, onReset, onRunTo, onRunToEvent }: Props) {
  const state = useSimStore((s) => s.state);
  const isPlaying = useSimStore((s) => s.isPlaying);
  const setPlaying = useSimStore((s) => s.setPlaying);
  const [runToInput, setRunToInput] = useState("");
  const [runToEventInput, setRunToEventInput] = useState("");

  const timeStr = state ? state.time_s.toFixed(4) : "0.0000";
  const eventsStr = state ? state.events_processed.toLocaleString() : "0";
  const heapStr = state ? state.heap_size.toLocaleString() : "0";

  const handleRunTo = () => {
    const t = parseFloat(runToInput);
    if (!isNaN(t) && t > (state?.time_s ?? 0)) {
      onRunTo(t);
    }
  };

  const handleRunToEvent = () => {
    const n = parseInt(runToEventInput, 10);
    if (!isNaN(n) && n > (state?.events_processed ?? 0)) {
      onRunToEvent(n);
    }
  };

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-gray-900 border-b border-gray-800 text-sm">
      <div className="flex items-center gap-3 text-gray-400">
        <span>
          t = <span className="text-white font-mono">{timeStr}s</span>
        </span>
        <span className="text-gray-700">|</span>
        <span>
          Events: <span className="text-white font-mono">{eventsStr}</span>
        </span>
        <span className="text-gray-700">|</span>
        <span>
          Heap: <span className="text-white font-mono">{heapStr}</span>
        </span>
      </div>

      <div className="flex-1" />

      <div className="flex items-center gap-2">
        <button
          onClick={() => onStep(1)}
          disabled={isPlaying || state?.is_complete}
          className="px-3 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 rounded text-xs font-medium"
          title="Step 1 event"
        >
          Step
        </button>

        <button
          onClick={() => onStep(10)}
          disabled={isPlaying || state?.is_complete}
          className="px-3 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 rounded text-xs font-medium"
          title="Step 10 events"
        >
          +10
        </button>

        {!isPlaying ? (
          <button
            onClick={() => {
              setPlaying(true);
              onPlay(10);
            }}
            disabled={state?.is_complete}
            className="px-3 py-1 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 rounded text-xs font-medium"
          >
            Play
          </button>
        ) : (
          <button
            onClick={() => {
              setPlaying(false);
              onPause();
            }}
            className="px-3 py-1 bg-amber-700 hover:bg-amber-600 rounded text-xs font-medium"
          >
            Pause
          </button>
        )}

        <select
          onChange={(e) => {
            const speed = Number(e.target.value);
            if (isPlaying) {
              onPlay(speed);
            }
          }}
          defaultValue={10}
          className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs"
          title="Events per batch"
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}/batch
            </option>
          ))}
        </select>

        <span className="text-gray-700">|</span>

        <div className="flex items-center gap-1">
          <input
            type="text"
            value={runToInput}
            onChange={(e) => setRunToInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleRunTo()}
            placeholder="time (s)"
            disabled={isPlaying || state?.is_complete}
            className="w-20 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white placeholder-gray-600 disabled:opacity-40"
          />
          <button
            onClick={handleRunTo}
            disabled={isPlaying || state?.is_complete || !runToInput}
            className="px-3 py-1 bg-blue-700 hover:bg-blue-600 disabled:opacity-40 rounded text-xs font-medium"
            title="Run simulation to specified time"
          >
            Run To
          </button>
        </div>

        <div className="flex items-center gap-1">
          <input
            type="text"
            value={runToEventInput}
            onChange={(e) => setRunToEventInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleRunToEvent()}
            placeholder="event #"
            disabled={isPlaying || state?.is_complete}
            className="w-20 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-white placeholder-gray-600 disabled:opacity-40"
          />
          <button
            onClick={handleRunToEvent}
            disabled={isPlaying || state?.is_complete || !runToEventInput}
            className="px-3 py-1 bg-blue-700 hover:bg-blue-600 disabled:opacity-40 rounded text-xs font-medium"
            title="Run simulation to specified event number"
          >
            Run To #
          </button>
        </div>

        <span className="text-gray-700">|</span>

        <button
          onClick={() => {
            setPlaying(false);
            onReset();
          }}
          className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs font-medium"
        >
          Restart
        </button>
      </div>
    </div>
  );
}
