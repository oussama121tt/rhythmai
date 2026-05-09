import React, { useState } from "react";
import { apiFetch } from "../utils/api";
import { C } from "../utils/constants";

export default function AdminLogin({ nav, onAdminLogin }) {
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setError("");
    setLoading(true);
    try {
      const data = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email: form.email, password: form.password }),
      });
      if (!data.user.is_admin) throw new Error("Not an admin account");
      localStorage.setItem("ra_token", data.access_token);
      onAdminLogin(data.user, data.access_token);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", background: C.bg, padding: 20 }}>
      <div style={{ width: "100%", maxWidth: 420, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14, padding: 24, boxShadow: C.shadowLg }}>
        <h2 style={{ marginTop: 0, marginBottom: 6, fontSize: 24, color: C.text }}>Admin Portal</h2>
        <p style={{ marginTop: 0, marginBottom: 16, color: C.textMuted }}>System administration login</p>
        <input
          style={{ width: "100%", boxSizing: "border-box", border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px", marginBottom: 10 }}
          placeholder="Admin email"
          value={form.email}
          onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))}
        />
        <input
          style={{ width: "100%", boxSizing: "border-box", border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px", marginBottom: 10 }}
          type="password"
          placeholder="Password"
          value={form.password}
          onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
        />
        {error && <div style={{ color: C.red, marginBottom: 10, fontSize: 13 }}>{error}</div>}
        <button
          style={{ width: "100%", border: "none", borderRadius: 8, padding: "10px 12px", background: "linear-gradient(135deg,#7c3aed,#a855f7)", color: "#fff", fontWeight: 700, cursor: "pointer", opacity: loading ? 0.7 : 1 }}
          onClick={submit}
          disabled={loading}
        >
          {loading ? "Authenticating..." : "Enter Admin Portal"}
        </button>
        <button
          style={{ width: "100%", marginTop: 10, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 12px", background: C.surface, color: C.text, fontWeight: 600, cursor: "pointer" }}
          onClick={() => nav("landing")}
        >
          Back to landing
        </button>
      </div>
    </div>
  );
}
