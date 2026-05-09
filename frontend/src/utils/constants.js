/**
 * Constants and styling for RhythmAI frontend
 * Includes color palette, class definitions, inline styles, and assets
 */

// ── Logo (embedded SVG) ──────────────────────────────────────────────────────
export const LOGO_DATA_URL =
  "data:image/svg+xml;charset=UTF-8," +
  encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180" role="img" aria-label="RhythmAI logo">
      <path d="M138 146c-35-20-70-47-85-80C36 38 49 15 74 9c18-4 34 4 43 19 10-15 26-23 43-19 25 6 38 29 23 58-16 32-50 59-85 79Z" fill="none" stroke="#d71f2c" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M58 92h26l12-20 15 44 15-56 13 28h21" fill="none" stroke="#d71f2c" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M150 92h28l12-18 14 22 11-22 11 18h15" fill="none" stroke="#1d4ed8" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M224 92h18l10-35 12 59 14-50 11 26h18" fill="none" stroke="#3da53b" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M266 92h19l8-18 10 35 10-27 6 10h12" fill="none" stroke="#f28c18" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M300 92h10" fill="none" stroke="#7c3aed" stroke-width="7" stroke-linecap="round"/>
      <path d="M137 11c14 0 24 8 29 18" fill="none" stroke="#0f172a" stroke-width="6" stroke-linecap="round"/>
      <path d="M162 8c11 0 19 4 24 12" fill="none" stroke="#0f172a" stroke-width="6" stroke-linecap="round"/>
      <path d="M185 11c9 0 16 3 20 10" fill="none" stroke="#0f172a" stroke-width="6" stroke-linecap="round"/>
    </svg>
  `);

// ── Palette (Light, professional) ────────────────────────────────────────────
export const C = {
  bg: "#f0f4f8",
  surface: "#ffffff",
  card: "#ffffff",
  sidebar: "#0f1f3d",
  sidebarHover: "#1a3358",
  sidebarActive: "#1e3a6e",
  border: "#dde3ef",
  accent: "#1a56db",
  accentLight: "#e8f0fe",
  green: "#0d9f6e",
  greenLight: "#dcfce7",
  red: "#e02424",
  redLight: "#fee2e2",
  yellow: "#d97706",
  yellowLight: "#fef3c7",
  purple: "#7c3aed",
  purpleLight: "#ede9fe",
  orange: "#ea580c",
  orangeLight: "#ffedd5",
  text: "#1a1f36",
  textMuted: "#6b7fa3",
  textLight: "#94a3b8",
  gradient: "linear-gradient(135deg,#1a56db 0%,#0ea5e9 100%)",
  gradientSoft: "linear-gradient(135deg,#e8f0fe 0%,#dbeafe 100%)",
  shadow: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)",
  shadowMd: "0 4px 16px rgba(0,0,0,0.08)",
  shadowLg: "0 8px 32px rgba(0,0,0,0.1)",
};

// ── Diagnostic Classes ────────────────────────────────────────────────────────
export const CLASSES = [
  { key: "NORM", label: "Normal", color: C.green, bgLight: C.greenLight, desc: "No pathology detected", icon: "✓" },
  { key: "MI",   label: "Myocardial Infarction", color: C.red, bgLight: C.redLight, desc: "Signs of heart muscle damage", icon: "⚠" },
  { key: "STTC", label: "ST/T Changes", color: C.yellow, bgLight: C.yellowLight, desc: "Repolarisation abnormalities", icon: "〰" },
  { key: "CD",   label: "Conduction Disturbance", color: C.purple, bgLight: C.purpleLight, desc: "Bundle branch / AV block", icon: "⚡" },
  { key: "ARR",  label: "Arrhythmia", color: C.orange, bgLight: C.orangeLight, desc: "Irregular heart rhythm", icon: "∿" },
];

export const colorFor = k => CLASSES.find(c => c.key === k)?.color || C.textMuted;
export const bgLightFor = k => CLASSES.find(c => c.key === k)?.bgLight || "#f1f5f9";

// ── Styles (CSS-in-JS) ────────────────────────────────────────────────────────
export const S = {
  app: { fontFamily: "'DM Sans', 'Inter', sans-serif", background: C.bg, minHeight: "100vh", color: C.text },
  // Sidebar
  sidebar: {
    position: "fixed", left: 0, top: 0, bottom: 0, width: 240,
    background: C.sidebar, zIndex: 300, display: "flex", flexDirection: "column",
    boxShadow: "2px 0 12px rgba(0,0,0,0.15)",
  },
  sidebarLogo: {
    padding: "20px 20px 16px", borderBottom: "1px solid rgba(255,255,255,0.08)",
    display: "flex", alignItems: "center", gap: 10,
  },
  sidebarLink: (active) => ({
    display: "flex", alignItems: "center", gap: 12, padding: "10px 16px 10px 20px",
    margin: "2px 8px", borderRadius: 8, cursor: "pointer", transition: "all .15s",
    background: active ? C.sidebarActive : "transparent",
    color: active ? "#ffffff" : "rgba(255,255,255,0.6)",
    fontSize: 18, fontWeight: active ? 600 : 400, textDecoration: "none",
    borderLeft: active ? `3px solid #60a5fa` : "3px solid transparent",
  }),
  sidebarSection: {
    fontSize: 15, fontWeight: 700, color: "rgba(255,255,255,0.3)",
    textTransform: "uppercase", letterSpacing: "0.08em",
    padding: "16px 20px 6px",
  },
  // Main content
  main: { marginLeft: 240, minHeight: "100vh", flex: 1, width: "calc(100% - 240px)", minWidth: 0 },
  topbar: {
    position: "sticky", top: 0, background: "rgba(240,244,248,0.9)",
    backdropFilter: "blur(10px)", borderBottom: `1px solid ${C.border}`,
    padding: "0 28px", height: 60, display: "flex", alignItems: "center",
    justifyContent: "space-between", zIndex: 200,
  },
  // Buttons
  btn: (v = "primary", sm = false, danger = false) => ({
    background: danger ? C.redLight : v === "primary" ? C.gradient : v === "ghost" ? "transparent" : C.surface,
    color: danger ? C.red : v === "primary" ? "#ffffff" : C.text,
    border: v === "outline" ? `1px solid ${C.border}` : danger ? `1px solid ${C.red}44` : "none",
    padding: sm ? "6px 14px" : "10px 20px",
    borderRadius: 8, cursor: "pointer", fontSize: sm ? 14 : 16,
    fontWeight: 600, transition: "all .15s", whiteSpace: "nowrap",
    boxShadow: v === "primary" ? "0 2px 8px rgba(26,86,219,0.25)" : "none",
  }),
  card: {
    background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
    padding: 20, boxShadow: C.shadow,
  },
  input: {
    background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8,
    padding: "10px 13px", color: C.text, fontSize: 16, width: "100%",
    outline: "none", boxSizing: "border-box", transition: "border .15s",
  },
  label: {
    fontSize: 14, color: C.textMuted, marginBottom: 5, display: "block",
    fontWeight: 600, letterSpacing: "0.02em",
  },
  tag: (color, bg) => ({
    display: "inline-flex", alignItems: "center", gap: 4,
    background: bg || `${color}15`, color,
    border: `1px solid ${color}30`,
    borderRadius: 20, padding: "2px 10px", fontSize: 11, fontWeight: 600,
  }),
};

// ── Navigation items ──────────────────────────────────────────────────────────
export const NAV_ITEMS = [
  { id: "dashboard",  icon: "⊞",  label: "Dashboard",     section: "main" },
  { id: "analysis",   icon: "📡", label: "Signal Analysis", section: "main" },
  { id: "history",    icon: "📋", label: "History",         section: "main" },
];
