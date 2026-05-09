import React from "react";
import Avatar from "./Avatar";
import NotificationBell from "../NotificationBell";
import { C, S } from "../utils/constants";

export default function Topbar({ title, subtitle, user }) {
  return (
    <div style={S.topbar}>
      <div>
        <div style={{ fontSize: 30, fontWeight: 700, color: C.text }}>{title}</div>
        {subtitle && <div style={{ fontSize: 14, color: C.textMuted }}>{subtitle}</div>}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {user && <NotificationBell apiFetch={() => {}} />}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            background: C.surface,
            border: `1px solid ${C.border}`,
            borderRadius: 20,
          }}
        >
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.green }} />
          <span style={{ fontSize: 20, color: C.textMuted }}>System Online</span>
        </div>
        <Avatar name={user?.name || "?"} size={36} color={C.accent} />
      </div>
    </div>
  );
}
