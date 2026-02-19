import { memo } from "react";

interface SlotsWidgetProps {
  total: number;
  active: number;
  color: string;
}

function SlotsWidget({ total, active, color }: SlotsWidgetProps) {
  const slots = Array.from({ length: total }, (_, i) => i < active);
  const cols = total > 8 ? Math.ceil(Math.sqrt(total)) : total;

  return (
    <div className="flex flex-col items-center gap-1.5 w-full">
      <div
        className="flex flex-wrap justify-center gap-1.5"
        style={{ maxWidth: cols * 20 }}
      >
        {slots.map((isActive, i) => (
          <div
            key={i}
            className="w-3.5 h-3.5 rounded-full border transition-colors duration-300"
            style={{
              borderColor: isActive ? color : "#4b5563",
              backgroundColor: isActive ? color : "transparent",
              boxShadow: isActive ? `0 0 6px ${color}60` : "none",
            }}
          />
        ))}
      </div>
      <div className="text-[10px] text-gray-400 font-mono">
        {active}/{total} active
      </div>
    </div>
  );
}

export default memo(SlotsWidget);
