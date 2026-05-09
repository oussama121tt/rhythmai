import React from "react";
import Avatar from "./Avatar";
import { C, S, LOGO_DATA_URL, NAV_ITEMS } from "../utils/constants";

export default function Sidebar({
  page,
  nav,
  user,
  onLogout,
  mobile = false,
  open = false,
  onClose = () => {},
  adminTabs = [],
  activeAdminTab = "overview",
  onAdminTabChange = () => {},
}) {
  const base = { ...S.sidebar };
  if (mobile) {
    Object.assign(base, {
      width: "80%",
      maxWidth: 320,
      transform: open ? "translateX(0)" : "translateX(-110%)",
      transition: "transform .22s ease",
      zIndex: 700,
    });
  }

  return (
    <>
      {mobile && open && (
        <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.25)", zIndex: 690 }} />
      )}

      <aside style={base}>
        <div style={S.sidebarLogo}>
          <img src={LOGO_DATA_URL} alt="RhythmAI" style={{ width: 38, height: 38, objectFit: "contain" }} />
          <div>
            <div style={{ fontSize: 20, fontWeight: 800, color: "#ffffff", letterSpacing: "-0.02em" }}>RhythmAI</div>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginTop: 1 }}>Cardiac Intelligence</div>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: "auto", paddingTop: 8 }}>
          {page !== "admin" ? (
            <>
              <div style={S.sidebarSection}>Navigation</div>
              {NAV_ITEMS.map((item) => (
                <div key={item.id} style={S.sidebarLink(page === item.id)} onClick={() => { nav(item.id); if (mobile) onClose(); }}>
                  <span style={{ fontSize: 18, width: 20, textAlign: "center" }}>{item.icon}</span>
                  <span>{item.label}</span>
                  {item.id === "analysis" && (
                    <span
                      style={{
                        marginLeft: "auto",
                        fontSize: 11,
                        background: "rgba(96,165,250,0.2)",
                        color: "#60a5fa",
                        padding: "2px 6px",
                        borderRadius: 10,
                        fontWeight: 700,
                      }}
                    >
                      AI
                    </span>
                  )}
                </div>
              ))}
            </>
          ) : (
            <>
              <div style={S.sidebarSection}>Admin Navigation</div>
              {adminTabs.map(([id, label]) => (
                <div
                  key={id}
                  style={S.sidebarLink(activeAdminTab === id)}
                  onClick={() => {
                    onAdminTabChange(id);
                    if (mobile) onClose();
                  }}
                >
                  <span style={{ fontSize: 18, width: 20, textAlign: "center" }}>{id === "overview" ? "◌" : id === "users" ? "👥" : "📈"}</span>
                  <span>{label}</span>
                </div>
              ))}
            </>
          )}
        </div>

        <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", padding: "12px 8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8 }}>
            <Avatar name={user?.name || "?"} size={34} color="#60a5fa" />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  color: "#fff",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {user?.name || "Doctor"}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: "rgba(255,255,255,0.4)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {user?.specialty || user?.role || ""}
              </div>
            </div>
            <button
              onClick={() => {
                onLogout?.();
                if (mobile) onClose();
              }}
              style={{
                background: "#e0242433",
                border: "1px solid #e0242466",
                color: "#ff6b6b",
                cursor: "pointer",
                fontSize: 15,
                padding: "6px 12px",
                borderRadius: 10,
                fontWeight: 500,
              }}
              title="Sign Out"
            >
              ⏻
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
