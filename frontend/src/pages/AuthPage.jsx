import React, { useState, useRef } from "react";
import { apiFetch } from "../utils/api";
import { C, S, LOGO_DATA_URL, CLASSES } from "../utils/constants";

export default function AuthPage({ mode, nav, onLogin }) {
  const isLogin = mode === "login";
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    confirm: "",
    role: "doctor",
    medical_id: "",
    specialty: "",
    hospital: "",
    idCardPhoto: null,
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const idCardRef = useRef(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const submit = async () => {
    setError("");
    setLoading(true);
    try {
      if (isLogin) {
        const data = await apiFetch("/api/auth/login", {
          method: "POST",
          body: JSON.stringify({ email: form.email, password: form.password }),
        });
        localStorage.setItem("ra_token", data.access_token);
        onLogin(data.user, data.access_token);
      } else {
        if (!form.name || !form.email || !form.password || !form.medical_id || !form.specialty || !form.hospital) {
          throw new Error("All fields are required");
        }
        if (form.password !== form.confirm) throw new Error("Passwords do not match");

        const fd = new FormData();
        Object.entries({
          name: form.name,
          email: form.email,
          password: form.password,
          role: form.role,
          medical_id: form.medical_id,
          specialty: form.specialty,
          hospital: form.hospital,
        }).forEach(([k, v]) => fd.append(k, v));
        if (form.idCardPhoto) fd.append("id_card_photo", form.idCardPhoto);

        await apiFetch("/api/auth/register", { method: "POST", body: fd });
        nav("login");
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, padding: 20, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ width: "100%", maxWidth: 1200, display: "flex", gap: 28, alignItems: "stretch" }}>
        <aside style={{ flex: 1, background: C.sidebar, color: "#fff", borderRadius: 16, padding: 36, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 18 }}>
          <div style={{ width: 140, height: 100, display: "grid", placeItems: "center", background: "rgba(255,255,255,0.06)", borderRadius: 14, padding: 8 }}>
            <img src={LOGO_DATA_URL} alt="RhythmAI" style={{ width: 120, height: 70, objectFit: "contain" }} />
          </div>
          <div style={{ fontSize: 40, fontWeight: 900, letterSpacing: "-0.04em", color: "#fff" }}>RhythmAI</div>
          <div style={{ color: "rgba(255,255,255,0.85)", textAlign: "center", maxWidth: 380, fontSize: 15 }}>AI-powered cardiac diagnostics for verified medical professionals</div>

          <svg viewBox="0 0 900 80" width="90%" height="80" style={{ marginTop: 8 }}>
            <defs>
              <linearGradient id="ecgWaveA" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#60a5fa" />
                <stop offset="100%" stopColor="#0ea5e9" />
              </linearGradient>
            </defs>
            <rect x="0" y="0" width="900" height="80" rx="10" fill="rgba(255,255,255,0.03)" />
            <path d="M0 40 H112 L132 40 L144 10 L156 70 L170 40 H232 L248 40 L262 30 L272 52 L286 40 H350 L370 40 L384 6 L398 70 L414 40 H488 L508 40 L522 30 L534 58 L548 40 H646 L668 40 L682 22 L694 72 L708 40 H900" fill="none" stroke="url(#ecgWaveA)" strokeWidth="3.2" strokeLinecap="round" />
          </svg>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 12 }}>
            {CLASSES.slice(0, 5).map((c) => (
              <div key={c.key} style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 14, color: "rgba(255,255,255,0.9)" }}>
                <div style={{ width: 10, height: 10, borderRadius: 10, background: c.color }} />
                <div>{c.label}</div>
              </div>
            ))}
          </div>
        </aside>

        <div style={{ width: 520, ...S.card, padding: 28 }}>
          <h2 style={{ marginTop: 0, marginBottom: 6, fontSize: 26 }}>{isLogin ? "Welcome back" : "Create account"}</h2>
          <p style={{ marginTop: 0, color: C.textMuted, marginBottom: 16 }}>{isLogin ? "Sign in to continue" : "Medical verification required"}</p>

          {!isLogin && (
            <>
              <label style={S.label}>Full Name</label>
              <input style={{ ...S.input, marginBottom: 10 }} value={form.name} onChange={set("name")} />
            </>
          )}
          <label style={S.label}>Email</label>
          <input style={{ ...S.input, marginBottom: 10 }} type="email" value={form.email} onChange={set("email")} />
          <label style={S.label}>Password</label>
          <input style={{ ...S.input, marginBottom: 10 }} type="password" value={form.password} onChange={set("password")} />

          {!isLogin && (
            <>
              <label style={S.label}>Confirm Password</label>
              <input style={{ ...S.input, marginBottom: 10 }} type="password" value={form.confirm} onChange={set("confirm")} />
              <label style={S.label}>Role</label>
              <select style={{ ...S.input, marginBottom: 10 }} value={form.role} onChange={set("role")}>
                <option value="doctor">Certified Doctor</option>
                <option value="resident">Resident / Intern</option>
                <option value="student">Medical Student</option>
              </select>
              <label style={S.label}>Medical ID</label>
              <input style={{ ...S.input, marginBottom: 10 }} value={form.medical_id} onChange={set("medical_id")} />
              <label style={S.label}>Specialty</label>
              <input style={{ ...S.input, marginBottom: 10 }} value={form.specialty} onChange={set("specialty")} />
              <label style={S.label}>Hospital</label>
              <input style={{ ...S.input, marginBottom: 10 }} value={form.hospital} onChange={set("hospital")} />
              <input
                ref={idCardRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setForm((prev) => ({ ...prev, idCardPhoto: f }));
                }}
              />
              <button style={{ ...S.btn("outline", true), marginBottom: 10 }} onClick={() => idCardRef.current?.click()}>
                {form.idCardPhoto ? `ID Card: ${form.idCardPhoto.name}` : "Upload ID Card (optional)"}
              </button>
            </>
          )}

          {error && <div style={{ color: C.red, fontSize: 13, marginBottom: 10 }}>{error}</div>}

          <button style={{ ...S.btn(), width: "100%", marginBottom: 10, opacity: loading ? 0.7 : 1 }} onClick={submit} disabled={loading}>
            {loading ? "Please wait..." : isLogin ? "Sign In" : "Create Account"}
          </button>

          <button style={{ ...S.btn("outline", true), width: "100%", marginBottom: 8 }} onClick={() => nav(isLogin ? "signup" : "login")}>
            {isLogin ? "Need an account? Sign up" : "Already have an account? Sign in"}
          </button>
          <button style={{ ...S.btn("outline", true), width: "100%" }} onClick={() => nav("landing")}>Back to landing</button>
        </div>
      </div>
    </div>
  );
}
