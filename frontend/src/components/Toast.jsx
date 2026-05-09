import React from "react";
import { useEffect } from "react";
import { C } from "../utils/constants";

export default function Toast({ msg, type = "info", onClose }) {
  const bg = type === "error" ? C.red : type === "success" ? C.green : C.accent;

  useEffect(() => {
    const t = setTimeout(() => onClose?.(), 4000);
    return () => clearTimeout(t);
  }, [onClose]);

  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        right: 24,
        zIndex: 999,
        background: C.surface,
        border: `1px solid ${bg}33`,
        borderLeft: `4px solid ${bg}`,
        borderRadius: 12,
        padding: "14px 18px",
        maxWidth: 340,
        boxShadow: C.shadowLg,
      }}
    >
      <div style={{ color: bg, fontWeight: 700, fontSize: 15 }}>
        {type === "error" ? "❌" : type === "success" ? "✅" : "ℹ️"} {msg}
      </div>
    </div>
  );
}
