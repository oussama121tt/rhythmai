import React from "react";
import { C, LOGO_DATA_URL } from "../utils/constants";

function LandingButton({ children, variant = "primary", onClick }) {
  const styles = {
    primary: {
      background: C.gradient,
      color: "#fff",
      border: "none",
      boxShadow: "0 10px 24px rgba(26,86,219,0.22)",
    },
    secondary: {
      background: "#fff",
      color: C.text,
      border: `1px solid ${C.border}`,
    },
    admin: {
      background: "#fff",
      color: C.purple,
      border: `1px solid ${C.purple}55`,
    },
  };

  return (
    <button
      onClick={onClick}
      style={{
        ...styles[variant],
        borderRadius: 14,
        padding: "13px 18px",
        cursor: "pointer",
        fontWeight: 800,
        fontSize: 15,
        letterSpacing: "-0.01em",
        transition: "transform .16s ease, box-shadow .16s ease, opacity .16s ease",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-1px)";
        e.currentTarget.style.opacity = "0.98";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.opacity = "1";
      }}
    >
      {children}
    </button>
  );
}

function ECGCard() {
  return (
    <div
      style={{
        width: "min(860px, 100%)",
        background: "linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,251,255,0.90))",
        border: `1px solid ${C.border}`,
        borderRadius: 26,
        padding: 18,
        boxShadow: "0 22px 60px rgba(15,23,42,0.09)",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "linear-gradient(90deg, transparent 0, transparent 24px, rgba(26,86,219,0.035) 25px), linear-gradient(transparent 0, transparent 24px, rgba(26,86,219,0.035) 25px)",
          backgroundSize: "25px 25px",
          opacity: 0.42,
          pointerEvents: "none",
        }}
      />
      <div style={{ position: "relative", zIndex: 1 }}>
        <svg viewBox="0 0 900 130" width="100%" height="130" role="img" aria-label="ECG waveform">
          <defs>
            <linearGradient id="ecgWave" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2563eb" />
              <stop offset="100%" stopColor="#0ea5e9" />
            </linearGradient>
            <linearGradient id="ecgGlowFill" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(14,165,233,0.32)" />
              <stop offset="100%" stopColor="rgba(14,165,233,0)" />
            </linearGradient>
            <filter id="ecgGlow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <rect x="0" y="0" width="900" height="130" rx="18" fill="rgba(255,255,255,0.40)" />
          <path d="M0 67 H112 L132 67 L144 40 L156 93 L170 67 H232 L248 67 L262 58 L272 78 L286 67 H350 L370 67 L384 31 L398 103 L414 67 H488 L508 67 L522 53 L534 82 L548 67 H646 L668 67 L682 46 L694 95 L708 67 H900" fill="none" stroke="rgba(37,99,235,0.16)" strokeWidth="16" strokeLinecap="round" />
          <path d="M0 67 H112 L132 67 L144 40 L156 93 L170 67 H232 L248 67 L262 58 L272 78 L286 67 H350 L370 67 L384 31 L398 103 L414 67 H488 L508 67 L522 53 L534 82 L548 67 H646 L668 67 L682 46 L694 95 L708 67 H900" fill="none" stroke="url(#ecgWave)" strokeWidth="4.2" strokeLinecap="round" strokeLinejoin="round" filter="url(#ecgGlow)" />
          <path d="M0 67 H112 L132 67 L144 40 L156 93 L170 67 H232 L248 67 L262 58 L272 78 L286 67 H350 L370 67 L384 31 L398 103 L414 67 H488 L508 67 L522 53 L534 82 L548 67 H646 L668 67 L682 46 L694 95 L708 67 H900" fill="none" stroke="url(#ecgGlowFill)" strokeWidth="12" opacity="0.45" />
        </svg>
      </div>
    </div>
  );
}

export default function Landing({ nav }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        position: "relative",
        overflow: "hidden",
        backgroundColor: "#eef4fb",
        backgroundImage:
          "linear-gradient(rgba(37,99,235,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(37,99,235,0.06) 1px, transparent 1px)",
        backgroundSize: "36px 36px",
        backgroundPosition: "center center",
      }}
    >
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(circle at 20% 20%, rgba(59,130,246,0.10), transparent 30%), radial-gradient(circle at 80% 20%, rgba(14,165,233,0.10), transparent 28%), radial-gradient(circle at 50% 80%, rgba(124,58,237,0.08), transparent 34%)" }} />

      <header
        style={{
          position: "relative",
          zIndex: 2,
          background: "rgba(255,255,255,0.93)",
          backdropFilter: "blur(16px)",
          borderBottom: `1px solid ${C.border}`,
        }}
      >
        <div style={{ maxWidth: 1400, margin: "0 auto", padding: "14px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <img src={LOGO_DATA_URL} alt="RhythmAI" style={{ width: 74, height: 44, objectFit: "contain", display: "block" }} />
            <div>
              <div style={{ fontSize: 29, fontWeight: 900, letterSpacing: "-0.05em", color: C.text }}>RhythmAI</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: C.textMuted, letterSpacing: "0.03em" }}>Cardiac Intelligence Platform</div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <LandingButton variant="secondary" onClick={() => nav("login")}>Sign In</LandingButton>
            <LandingButton variant="primary" onClick={() => nav("signup")}>Request Access →</LandingButton>
            <LandingButton variant="admin" onClick={() => nav("admin-login")}>Admin</LandingButton>
          </div>
        </div>
      </header>

      <main style={{ position: "relative", zIndex: 2, minHeight: "calc(100vh - 86px)", display: "grid", placeItems: "center", padding: "20px 24px 24px", overflow: "hidden" }}>
        <div style={{ maxWidth: 1260, width: "100%", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: 18 }}>
          <div style={{ width: 150, height: 102, display: "grid", placeItems: "center", borderRadius: 28, background: "rgba(255,255,255,0.76)", border: `1px solid ${C.border}`, boxShadow: "0 22px 50px rgba(15,23,42,0.08)", padding: 6, backdropFilter: "blur(10px)" }}>
            <img src={LOGO_DATA_URL} alt="RhythmAI" style={{ width: 138, height: 78, objectFit: "contain", display: "block" }} />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "center" }}>
            <div style={{ fontSize: "clamp(48px, 6.2vw, 76px)", lineHeight: 0.9, fontWeight: 900, letterSpacing: "-0.07em", color: "#0f172a", fontFamily: "Georgia, 'Times New Roman', serif" }}>RhythmAI</div>
            <div
              style={{
                fontSize: "clamp(36px, 4.8vw, 60px)",
                lineHeight: 0.9,
                fontWeight: 900,
                letterSpacing: "-0.05em",
                background: "linear-gradient(135deg,#0f3d91 0%,#1d4ed8 40%,#0ea5e9 100%)",
                WebkitBackgroundClip: "text",
                backgroundClip: "text",
                color: "transparent",
              }}
            >
              Cardiac Analysis
            </div>
          </div>

          <p style={{ maxWidth: 920, margin: 0, color: "#6780a6", fontSize: "clamp(15px, 1.4vw, 18px)", lineHeight: 1.5, fontWeight: 500 }}>
            Automatic classification of cardiac pathologies from ECG signals and images. Built for verified cardiologists, powered by GatedFusion deep learning.
          </p>

          <ECGCard />

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
            <LandingButton variant="primary" onClick={() => nav("signup")}>Request Access →</LandingButton>
            <LandingButton variant="secondary" onClick={() => nav("login")}>Sign In</LandingButton>
          </div>
        </div>
      </main>
    </div>
  );
}
