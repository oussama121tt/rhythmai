import React from "react";
import { C } from "../utils/constants";

export default function Modal({ onClose, width = 560, children }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(15,23,42,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: 16,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "100%",
          maxWidth: width,
          maxHeight: "90vh",
          overflowY: "auto",
          background: C.surface,
          border: `1px solid ${C.border}`,
          borderRadius: 14,
          boxShadow: C.shadowLg,
          padding: 20,
          position: "relative",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            width: 28,
            height: 28,
            borderRadius: 8,
            border: `1px solid ${C.border}`,
            background: C.surface,
            cursor: "pointer",
            color: C.textMuted,
            fontWeight: 700,
          }}
        >
          ×
        </button>
        {children}
      </div>
    </div>
  );
}
