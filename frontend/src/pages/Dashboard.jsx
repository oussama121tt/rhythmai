import React, { useState, useEffect } from "react";
import { apiFetch } from "../utils/api";
import { C, S, CLASSES, colorFor, bgLightFor } from "../utils/constants";
import Topbar from "../components/Topbar";
import Avatar from "../components/Avatar";
import ECGLine from "../components/ECGLine";
import DonutChart from "../components/DonutChart";
import Sparkline from "../components/Sparkline";
import StatusBadge from "../components/StatusBadge";

export default function Dashboard({ user, nav, analyses = [] }) {
  const [serverAlive, setServerAlive] = useState(true);

  useEffect(() => {
    let active = true;
    apiFetch("/api/health")
      .then(() => active && setServerAlive(true))
      .catch(() => active && setServerAlive(false));
    return () => {
      active = false;
    };
  }, []);

  const classCounts = CLASSES.map((c) => analyses.filter((a) => a.result === c.key).length);
  const recent = analyses.slice(0, 5);

  return (
    <>
      <Topbar title="Dashboard" subtitle={`Welcome back, ${user?.name || "Doctor"}`} user={user} />
      <div style={{ padding: 28 }}>
        <div style={{ ...S.card, marginBottom: 18, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <Avatar name={user?.name || "?"} size={44} color={C.accent} />
            <div>
              <div style={{ fontSize: 20, fontWeight: 800 }}>{user?.name || "User"}</div>
              <div style={{ color: C.textMuted, fontSize: 14 }}>{user?.specialty || user?.role || "Cardiology"}</div>
            </div>
            <StatusBadge status={user?.status || "verified"} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ color: serverAlive ? C.green : C.red, fontWeight: 700, fontSize: 13 }}>
              {serverAlive ? "API Online" : "API Unreachable"}
            </div>
            <ECGLine width={160} height={32} color={C.accent} />
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: 14, marginBottom: 18 }}>
          <div style={S.card}>
            <div style={{ fontSize: 12, color: C.textMuted, textTransform: "uppercase", fontWeight: 700 }}>Total Analyses</div>
            <div style={{ fontSize: 38, fontWeight: 900, color: C.accent }}>{analyses.length}</div>
          </div>
          <div style={S.card}>
            <div style={{ fontSize: 12, color: C.textMuted, textTransform: "uppercase", fontWeight: 700 }}>Critical MI/ARR</div>
            <div style={{ fontSize: 38, fontWeight: 900, color: C.red }}>
              {analyses.filter((a) => ["MI", "ARR"].includes(a.result)).length}
            </div>
          </div>
          <div style={S.card}>
            <div style={{ fontSize: 12, color: C.textMuted, textTransform: "uppercase", fontWeight: 700 }}>Average Confidence</div>
            <div style={{ fontSize: 38, fontWeight: 900, color: C.green }}>
              {analyses.length ? `${Math.round(analyses.reduce((s, a) => s + (a.confidence || 0), 0) / analyses.length)}%` : "—"}
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 18 }}>
          <div style={S.card}>
            <div style={{ marginBottom: 10, fontSize: 16, fontWeight: 700 }}>Diagnosis Distribution</div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <DonutChart values={classCounts} colors={CLASSES.map((c) => c.color)} size={110} />
              <div style={{ flex: 1 }}>
                {CLASSES.map((c) => (
                  <div key={c.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6, fontSize: 14 }}>
                    <span style={{ color: c.color, fontWeight: 700 }}>{c.key}</span>
                    <span style={{ color: C.textMuted }}>{analyses.filter((a) => a.result === c.key).length}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div style={S.card}>
            <div style={{ marginBottom: 10, fontSize: 16, fontWeight: 700 }}>Activity Trend</div>
            <Sparkline data={analyses.slice(0, 30).map((a) => a.confidence || 0).reverse()} color={C.accent} width={320} height={70} />
          </div>
        </div>

        <div style={{ ...S.card, marginBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <div style={{ fontSize: 16, fontWeight: 700 }}>Recent Analyses</div>
            <div style={{ display: "flex", gap: 8 }}>
              <button style={S.btn("outline", true)} onClick={() => nav("analysis")}>New Analysis</button>
              <button style={S.btn("outline", true)} onClick={() => nav("history")}>View History</button>
            </div>
          </div>
          {recent.length === 0 && <div style={{ color: C.textMuted, fontSize: 14 }}>No analyses yet.</div>}
          {recent.map((r) => (
            <div key={r.id || r.ecg_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: `1px solid ${C.border}` }}>
              <span style={{ width: 110, fontFamily: "monospace", color: C.accent }}>{r.ecg_id}</span>
              <span style={S.tag(colorFor(r.result), bgLightFor(r.result))}>{r.result}</span>
              <span style={{ color: C.textMuted }}>{r.created_at?.slice(0, 10)}</span>
              <span style={{ marginLeft: "auto", color: C.textMuted }}>{Math.round(r.confidence || 0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
