import React from "react";
import { C, S } from "../utils/constants";

export default function StatusBadge({ status }) {
  const map = {
    verified: [C.green, C.greenLight, "✓ Verified"],
    pending: [C.yellow, C.yellowLight, "⏳ Pending"],
    suspended: [C.red, C.redLight, "⛔ Suspended"],
  };
  const [color, bg, label] = map[status] || [C.textMuted, C.bg, status || "Unknown"];
  return <span style={S.tag(color, bg)}>{label}</span>;
}
