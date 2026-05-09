import React, { useState, useEffect, useCallback } from "react";
import { apiFetch } from "../utils/api";
import { C, S, CLASSES, colorFor, bgLightFor } from "../utils/constants";
import Sidebar from "../components/Sidebar";
import Avatar from "../components/Avatar";
import Modal from "../components/Modal";
import StatusBadge from "../components/StatusBadge";
import DonutChart from "../components/DonutChart";
import Sparkline from "../components/Sparkline";
import AnalysisPage from "./AnalysisPage";

export default function AdminDashboard({ nav, onLogout, toast = () => {} }) {
  const [tab, setTab] = useState("overview");
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [allAnalyses, setAllAnalyses] = useState([]);
  const [search, setSearch] = useState("");
  const [userModal, setUserModal] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, u, a] = await Promise.all([apiFetch("/api/admin/stats"), apiFetch("/api/admin/users"), apiFetch("/api/admin/analyses")]);
      setStats(s);
      setUsers(u);
      setAllAnalyses(a);
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    load();
  }, [load]);

  const updateUser = async (uid, patch) => {
    try {
      const updated = await apiFetch(`/api/admin/users/${uid}`, { method: "PATCH", body: JSON.stringify(patch) });
      setUsers((prev) => prev.map((u) => (u.id === uid ? updated : u)));
      if (userModal?.id === uid) setUserModal(updated);
      toast("User updated", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  };

  const filteredUsers = users.filter((u) =>
    [u.name, u.email, u.medical_id].join(" ").toLowerCase().includes(search.toLowerCase())
  );

  const adminTabs = [["overview", "Overview"], ["users", "Users"], ["analyses", "Analyses"], ["analysis", "ECG Analysis"]];
  const dailyArr = stats ? Object.values(stats.daily_analyses || {}) : [];

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: C.bg }}>
      <Sidebar
        page="admin"
        nav={nav}
        user={{ name: "Admin", role: "administrator" }}
        onLogout={onLogout}
        adminTabs={adminTabs}
        activeAdminTab={tab}
        onAdminTabChange={setTab}
      />
      <div style={{ marginLeft: 240, flex: 1, padding: 20 }}>
        {userModal && (
          <Modal onClose={() => setUserModal(null)} width={560}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <Avatar name={userModal.name} size={46} color={C.purple} />
              <div>
                <div style={{ fontSize: 18, fontWeight: 800 }}>{userModal.name}</div>
                <div style={{ color: C.textMuted, fontSize: 13 }}>{userModal.email}</div>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
              {[["Medical ID", userModal.medical_id], ["Hospital", userModal.hospital || "—"], ["Specialty", userModal.specialty || "—"], ["Status", userModal.status]].map(([k, v]) => (
                <div key={k} style={{ ...S.card, padding: 12, background: C.bg }}>
                  <div style={{ fontSize: 10, color: C.textMuted, textTransform: "uppercase", fontWeight: 700 }}>{k}</div>
                  <div style={{ fontSize: 14, fontWeight: 700 }}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <button style={S.btn("outline", true)} onClick={() => updateUser(userModal.id, { status: "verified" })}>Verify</button>
              <button style={S.btn("outline", true)} onClick={() => updateUser(userModal.id, { status: "pending" })}>Pending</button>
              <button style={S.btn("outline", true)} onClick={() => updateUser(userModal.id, { status: "suspended" })}>Suspend</button>
            </div>
          </Modal>
        )}

        <div style={{ ...S.card, marginBottom: 14, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 24, fontWeight: 800 }}>Admin Portal</div>
            <div style={{ color: C.textMuted, fontSize: 13 }}>System Administration</div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {adminTabs.map(([k, l]) => (
              <button key={k} style={S.btn(tab === k ? "primary" : "outline", true)} onClick={() => setTab(k)}>{l}</button>
            ))}
          </div>
        </div>

        {loading && <div style={{ color: C.textMuted }}>Loading...</div>}

        {!loading && tab === "analysis" && (
          <AnalysisPage
            user={{ ...userModal, ...{ name: "Admin", role: "administrator", is_admin: true } }}
            onNewAnalysis={() => load()}
            toast={toast}
          />
        )}

        {!loading && tab === "overview" && stats && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 12, marginBottom: 14 }}>
              {[
                ["Users", stats.total_users, C.accent],
                ["Pending", stats.status_dist?.pending || 0, C.yellow],
                ["Suspended", stats.status_dist?.suspended || 0, C.red],
                ["Analyses", stats.total_analyses, C.green],
              ].map(([k, v, col]) => (
                <div key={k} style={{ ...S.card, borderColor: `${col}33` }}>
                  <div style={{ fontSize: 12, color: C.textMuted, textTransform: "uppercase", fontWeight: 700 }}>{k}</div>
                  <div style={{ fontSize: 36, fontWeight: 900, color: col }}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={S.card}>
                <div style={{ marginBottom: 10, fontSize: 16, fontWeight: 700 }}>Diagnosis Distribution</div>
                <DonutChart values={CLASSES.map((c) => stats.class_dist?.[c.key] || 0)} colors={CLASSES.map((c) => c.color)} size={100} />
              </div>
              <div style={S.card}>
                <div style={{ marginBottom: 10, fontSize: 16, fontWeight: 700 }}>Activity (30 days)</div>
                <Sparkline data={dailyArr.length ? dailyArr : [0, 0]} color={C.accent} width={320} height={70} />
              </div>
            </div>
          </>
        )}

        {!loading && tab === "users" && (
          <div style={S.card}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <div style={{ fontSize: 16, fontWeight: 700 }}>User Management</div>
              <input style={{ ...S.input, width: 280 }} placeholder="Search user" value={search} onChange={(e) => setSearch(e.target.value)} />
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: `2px solid ${C.border}` }}>
                  {["User", "Medical ID", "Role", "Status", "Joined"].map((h) => (
                    <th key={h} style={{ textAlign: "left", padding: "10px 12px", color: C.textMuted, fontSize: 11, textTransform: "uppercase" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((u) => (
                  <tr key={u.id} style={{ borderBottom: `1px solid ${C.border}`, cursor: "pointer" }} onClick={() => setUserModal(u)}>
                    <td style={{ padding: "10px 12px" }}>{u.name}</td>
                    <td style={{ padding: "10px 12px", fontFamily: "monospace" }}>{u.medical_id}</td>
                    <td style={{ padding: "10px 12px", textTransform: "capitalize" }}>{u.role}</td>
                    <td style={{ padding: "10px 12px" }}><StatusBadge status={u.status} /></td>
                    <td style={{ padding: "10px 12px", color: C.textMuted }}>{u.joined_at?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!loading && tab === "analyses" && (
          <div style={S.card}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>All Analyses ({allAnalyses.length})</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: `2px solid ${C.border}` }}>
                  {["ECG ID", "Doctor", "Date", "Patient", "Result", "Confidence"].map((h) => (
                    <th key={h} style={{ textAlign: "left", padding: "10px 12px", color: C.textMuted, fontSize: 11, textTransform: "uppercase" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {allAnalyses.map((r) => (
                  <tr key={r.id} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: "10px 12px", fontFamily: "monospace", color: C.accent }}>{r.ecg_id}</td>
                    <td style={{ padding: "10px 12px" }}>{r.doctor_name || "Unknown"}</td>
                    <td style={{ padding: "10px 12px", color: C.textMuted }}>{r.created_at?.slice(0, 10)}</td>
                    <td style={{ padding: "10px 12px" }}>{r.patient_ref || "—"}</td>
                    <td style={{ padding: "10px 12px" }}><span style={S.tag(colorFor(r.result), bgLightFor(r.result))}>{r.result}</span></td>
                    <td style={{ padding: "10px 12px" }}>{Math.round(r.confidence || 0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
