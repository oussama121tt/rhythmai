import React, { useState } from "react";
import { CLASSES, C, S, colorFor, bgLightFor } from "../utils/constants";
import Modal from "./Modal";

export default function AnalysisDetailModal({ analysis, onClose, onUpdate, onDelete }) {
  const [notes, setNotes] = useState(analysis.notes || "");
  const [editing, setEditing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const scores = CLASSES.map((c) => analysis.scores?.[c.key] ?? 0);
  const pc = CLASSES.find((c) => c.key === analysis.result);

  return (
    <Modal onClose={onClose} width={640}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20, paddingRight: 28 }}>
        <div>
          <div style={{ fontFamily: "monospace", color: C.accent, fontSize: 18, marginBottom: 4 }}>{analysis.ecg_id}</div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 800 }}>{analysis.created_at?.slice(0, 10)}</h2>
        </div>
        <span style={S.tag(colorFor(analysis.result), bgLightFor(analysis.result))}>{analysis.result} · {pc?.label}</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
        {[
          ["Patient Ref", analysis.patient_ref || "—"],
          ["Input Mode", analysis.input_mode],
          ["Age", analysis.patient_age || "—"],
          ["Sex", analysis.patient_sex || "—"],
          ["Inference", `${analysis.inference_time}s`],
          ["Confidence", `${Math.round(analysis.confidence)}%`],
        ].map(([k, v]) => (
          <div key={k} style={{ ...S.card, padding: 14, background: C.bg }}>
            <div style={{ fontSize: 10, color: C.textMuted, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".04em", marginBottom: 4 }}>{k}</div>
            <div style={{ fontSize: 15, fontWeight: 700 }}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{ ...S.card, marginBottom: 14, background: C.bg }}>
        <h3 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 700 }}>Probability Breakdown</h3>
        {CLASSES.map((c, i) => (
          <div key={c.key} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <div style={{ width: 42, fontWeight: 700, color: c.color, fontSize: 15 }}>{c.key}</div>
            <div style={{ flex: 1, height: 6, background: C.border, borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${scores[i]}%`, height: "100%", background: c.color, borderRadius: 3 }} />
            </div>
            <div style={{ width: 38, textAlign: "right", fontSize: 11, color: C.textMuted }}>{scores[i]}%</div>
          </div>
        ))}
      </div>

      <div style={{ ...S.card, marginBottom: 14, background: C.bg }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>📝 Clinical Notes</h3>
          {!editing ? (
            <button style={S.btn("outline", true)} onClick={() => setEditing(true)}>Edit</button>
          ) : (
            <div style={{ display: "flex", gap: 8 }}>
              <button style={S.btn("primary", true)} onClick={async () => { await onUpdate(analysis.id, notes); setEditing(false); }}>Save</button>
              <button style={S.btn("outline", true)} onClick={() => { setNotes(analysis.notes || ""); setEditing(false); }}>Cancel</button>
            </div>
          )}
        </div>
        {editing ? (
          <textarea
            style={{ ...S.input, minHeight: 90, resize: "vertical", lineHeight: 1.6 }}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add clinical observations..."
          />
        ) : (
          <div style={{ fontSize: 13, color: notes ? C.text : C.textMuted, lineHeight: 1.7, minHeight: 36 }}>
            {notes || "No notes yet. Click Edit to add."}
          </div>
        )}
      </div>

      {!confirmDelete ? (
        <button style={{ ...S.btn("outline", true), color: C.red, borderColor: `${C.red}40` }} onClick={() => setConfirmDelete(true)}>🗑 Delete Analysis</button>
      ) : (
        <div style={{ background: C.redLight, borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 13, color: C.red, fontWeight: 700, marginBottom: 10 }}>Delete this analysis? Cannot be undone.</div>
          <div style={{ display: "flex", gap: 8 }}>
            <button style={{ ...S.btn("primary", true), background: C.red }} onClick={() => onDelete(analysis.id)}>Yes, Delete</button>
            <button style={S.btn("outline", true)} onClick={() => setConfirmDelete(false)}>Cancel</button>
          </div>
        </div>
      )}
    </Modal>
  );
}
