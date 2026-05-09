import React from "react";
import { C } from "../utils/constants";

export default function DonutChart({ values = [], colors = [], size = 120, centerLabel = "" }) {
  const total = values.reduce((a, b) => a + b, 0) || 1;
  let cum = -Math.PI / 2;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 6;
  const inner = r * 0.62;

  const slices = values.map((v, i) => {
    const a = (v / total) * 2 * Math.PI;
    const x1 = cx + r * Math.cos(cum);
    const y1 = cy + r * Math.sin(cum);
    const x2 = cx + r * Math.cos(cum + a);
    const y2 = cy + r * Math.sin(cum + a);
    const xi1 = cx + inner * Math.cos(cum);
    const yi1 = cy + inner * Math.sin(cum);
    const xi2 = cx + inner * Math.cos(cum + a);
    const yi2 = cy + inner * Math.sin(cum + a);
    const big = a > Math.PI ? 1 : 0;
    const d = `M${xi1},${yi1}L${x1},${y1}A${r},${r} 0 ${big},1 ${x2},${y2}L${xi2},${yi2}A${inner},${inner} 0 ${big},0 ${xi1},${yi1}`;
    cum += a;
    return <path key={i} d={d} fill={colors[i]} opacity={0.9} />;
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {slices}
      <circle cx={cx} cy={cy} r={inner} fill="white" />
      {centerLabel && (
        <text
          x={cx}
          y={cy}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={C.text}
          fontSize={size * 0.12}
          fontWeight="bold"
        >
          {centerLabel}
        </text>
      )}
    </svg>
  );
}
