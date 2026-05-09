import React from "react";
import { C, CLASSES } from "../utils/constants";
import AIChatPanel from "../components/AIChatPanel";

export default function ResultWithChat({ result, onReset, patientRef, age, sex }) {
  const scores = CLASSES.map((c) => result.scores?.[c.key] ?? 0);
  const max = Math.max(...scores);
  const pc = CLASSES.find((c) => c.key === result.result);
  const patCtx = [patientRef && `Patient: ${patientRef}`, age && `Age: ${age}`, sex && `Sex: ${sex}`].filter(Boolean).join(", ");

  return (
    <div style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
      <AIChatPanel pathology={result.result} patientContext={patCtx} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ border: `1px solid ${C.border}`, borderTop: `4px solid ${pc?.color || C.accent}`, borderRadius: 12, background: C.surface, padding: 18, marginBottom: 14 }}>
          <div style={{ fontSize: 13, color: C.textMuted, fontWeight: 700, textTransform: "uppercase" }}>Primary Diagnosis</div>
          <div style={{ fontSize: 34, fontWeight: 900, color: pc?.color || C.text, lineHeight: 1.1, marginTop: 4 }}>{pc?.key || result.result}</div>
          <div style={{ fontSize: 16, color: C.textMuted, marginTop: 4 }}>{pc?.label || "Unknown class"}</div>
          <div style={{ marginTop: 10, fontSize: 13, color: C.textMuted }}>Inference: {result.inference_time}s</div>
        </div>

        <div style={{ border: `1px solid ${C.border}`, borderRadius: 12, background: C.surface, padding: 18, marginBottom: 14 }}>
          <h3 style={{ margin: "0 0 14px", fontSize: 16, fontWeight: 700 }}>Class Probability Breakdown</h3>
          {CLASSES.map((c, i) => (
            <div key={c.key} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
              <span style={{ width: 40, fontWeight: 700, color: c.color }}>{c.key}</span>
              <div style={{ flex: 1, height: 8, background: C.bg, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${scores[i]}%`, height: "100%", background: c.color, borderRadius: 4 }} />
              </div>
              <span style={{ width: 42, textAlign: "right", color: scores[i] === max ? c.color : C.textMuted }}>{scores[i]}%</span>
            </div>
          ))}
        </div>

        <div style={{ background: C.yellowLight, border: `1px solid ${C.yellow}44`, borderRadius: 10, padding: "12px 14px", marginBottom: 12, color: C.textMuted }}>
          RhythmAI provides decision support only. Final interpretation must be validated by a certified cardiologist.
        </div>

        <button style={{ border: "none", borderRadius: 8, padding: "10px 14px", background: C.gradient, color: "#fff", fontWeight: 700, cursor: "pointer" }} onClick={onReset}>New Analysis</button>
      </div>
    </div>
  );
}
