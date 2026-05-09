import React, { useState, useEffect } from "react";
import { apiFetch } from "../utils/api";
import { C, S, CLASSES, colorFor, bgLightFor } from "../utils/constants";
import Topbar from "../components/Topbar";
import StatusBadge from "../components/StatusBadge";
import AnalysisDetailModal from "../components/AnalysisDetailModal";

export default function HistoryPage({ analyses = [], onRefresh = () => {}, toast = () => {}, user = null }) {
  const [search, setSearch] = useState("");
  const [filterResult, setFilterResult] = useState("all");
  const [selected, setSelected] = useState(null);

  useEffect(() => {}, []);

  const filtered = analyses.filter((r) => {
    const q = search.toLowerCase();
    return (
      (r.ecg_id?.toLowerCase().includes(q) || r.result?.toLowerCase().includes(q) || (r.patient_ref || "").toLowerCase().includes(q)) &&
      (filterResult === "all" || r.result === filterResult)
    );
  });

  return (
    <>
      <Topbar title="Consultation History" subtitle={`${analyses.length} analyses on record`} user={user} />
      <div style={{ padding: 28 }}>
        {selected && (
          <AnalysisDetailModal
            analysis={selected}
            onClose={() => setSelected(null)}
            onUpdate={async (id, notes) => {
              try {
                await apiFetch(`/api/analyses/${id}/notes`, { method: "PATCH", body: JSON.stringify({ notes }) });
                toast("Notes saved", "success");
                onRefresh();
                setSelected((a) => (a ? { ...a, notes } : null));
              } catch (e) {
                toast(e.message, "error");
              }
            }}
            onDelete={async (id) => {
              try {
                await apiFetch(`/api/analyses/${id}`, { method: "DELETE" });
                toast("Deleted", "success");
                setSelected(null);
                onRefresh();
              } catch (e) {
                toast(e.message, "error");
              }
            }}
          />
        )}

        <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
          <button
            style={{ ...S.tag(filterResult === "all" ? C.accent : C.textMuted, filterResult === "all" ? C.accentLight : C.bg), cursor: "pointer", padding: "5px 14px", fontSize: 14 }}
            onClick={() => setFilterResult("all")}
          >
            All ({analyses.length})
          </button>
          {CLASSES.map((c) => {
            const count = analyses.filter((r) => r.result === c.key).length;
            return (
              <button
                key={c.key}
                style={{ ...S.tag(c.color, filterResult === c.key ? c.bgLight : C.surface), cursor: "pointer", padding: "5px 14px", fontSize: 14, border: `1px solid ${c.color}30` }}
                onClick={() => setFilterResult(filterResult === c.key ? "all" : c.key)}
              >
                {c.key} ({count})
              </button>
            );
          })}
          <input
            style={{ ...S.input, width: 260, marginLeft: "auto" }}
            placeholder="Search by ID, diagnosis, patient..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div style={S.card}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${C.border}` }}>
                {["ECG ID", "Date", "Patient", "Mode", "Result", "Confidence", "Status", ""].map((h) => (
                  <th key={h} style={{ textAlign: "left", padding: "10px 12px", color: C.textMuted, fontWeight: 700, fontSize: 11, textTransform: "uppercase" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.id} style={{ borderBottom: `1px solid ${C.border}`, cursor: "pointer" }} onClick={() => setSelected(r)}>
                  <td style={{ padding: "10px 12px", fontFamily: "monospace", color: C.accent }}>{r.ecg_id}</td>
                  <td style={{ padding: "10px 12px", color: C.textMuted }}>{r.created_at?.slice(0, 10)}</td>
                  <td style={{ padding: "10px 12px" }}>{r.patient_ref || "—"}</td>
                  <td style={{ padding: "10px 12px" }}><span style={S.tag(C.textMuted, C.bg)}>{r.input_mode}</span></td>
                  <td style={{ padding: "10px 12px" }}><span style={S.tag(colorFor(r.result), bgLightFor(r.result))}>{r.result}</span></td>
                  <td style={{ padding: "10px 12px" }}>{Math.round(r.confidence || 0)}%</td>
                  <td style={{ padding: "10px 12px" }}><StatusBadge status={r.status || "verified"} /></td>
                  <td style={{ padding: "10px 12px" }}>
                    <button style={S.btn("outline", true)} onClick={(e) => { e.stopPropagation(); setSelected(r); }}>View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && <div style={{ textAlign: "center", color: C.textMuted, padding: 28 }}>No analyses found</div>}
        </div>
      </div>
    </>
  );
}
