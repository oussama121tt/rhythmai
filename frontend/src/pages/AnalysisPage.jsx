import React, { useState, useRef } from "react";
import { apiFetch } from "../utils/api";
import { C, S, CLASSES } from "../utils/constants";
import Topbar from "../components/Topbar";
import AIChatPanel from "../components/AIChatPanel";

export default function AnalysisPage({ user, onNewAnalysis, toast = () => {} }) {
  const [tab, setTab] = useState("signal");
  const [signalFile, setSignalFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [patientRef, setPatientRef] = useState("");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("M");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const signalRef = useRef(null);
  const imageRef = useRef(null);

  const reset = () => {
    setSignalFile(null);
    setImageFile(null);
    setPatientRef("");
    setAge("");
    setSex("M");
    setResult(null);
  };

  const analyze = async () => {
    if ((tab === "signal" || tab === "fusion") && !signalFile) return toast("Please upload a signal file", "error");
    if ((tab === "image" || tab === "fusion") && !imageFile) return toast("Please upload an ECG image", "error");

    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("mode", tab);
      if (patientRef) fd.append("patient_ref", patientRef);
      if (age) fd.append("patient_age", age);
      fd.append("patient_sex", sex);
      if (signalFile) fd.append("signal_file", signalFile);
      if (imageFile) fd.append("image_file", imageFile);

      const data = await apiFetch("/api/analyses/predict", { method: "POST", body: fd });
      setResult(data);
      onNewAnalysis?.(data);
      toast("Analysis complete", "success");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    ["signal", "Signal"],
    ["image", "Image"],
    ["fusion", "Fusion"],
  ];

  return (
    <>
      <Topbar title="Signal Analysis" subtitle="Upload ECG data for AI-powered cardiac classification" user={user} />
      <div style={{ padding: 28, display: "flex", gap: 20, alignItems: "flex-start" }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
            {tabs.map(([k, l]) => (
              <button key={k} style={{ ...S.btn(tab === k ? "primary" : "outline", true) }} onClick={() => setTab(k)}>{l}</button>
            ))}
          </div>

          {!result ? (
            <>
              <div style={{ ...S.card, marginBottom: 12 }}>
                <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>Patient Information</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 90px", gap: 10 }}>
                  <input style={S.input} placeholder="Patient Ref" value={patientRef} onChange={(e) => setPatientRef(e.target.value)} />
                  <input style={S.input} placeholder="Age" type="number" value={age} onChange={(e) => setAge(e.target.value)} />
                  <select style={S.input} value={sex} onChange={(e) => setSex(e.target.value)}>
                    <option>M</option>
                    <option>F</option>
                  </select>
                </div>
              </div>

              <div style={{ ...S.card, marginBottom: 12 }}>
                <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>ECG Inputs</div>
                {(tab === "signal" || tab === "fusion") && (
                  <>
                    <input
                      ref={signalRef}
                      type="file"
                      accept=".dat,.npy,.csv"
                      style={{ display: "none" }}
                      onChange={(e) => setSignalFile(e.target.files?.[0] || null)}
                    />
                    <button style={{ ...S.btn("outline", true), marginBottom: 10 }} onClick={() => signalRef.current?.click()}>
                      {signalFile ? `Signal: ${signalFile.name}` : "Upload Signal (.dat/.npy/.csv)"}
                    </button>
                  </>
                )}

                {(tab === "image" || tab === "fusion") && (
                  <>
                    <input
                      ref={imageRef}
                      type="file"
                      accept=".png,.jpg,.jpeg"
                      style={{ display: "none" }}
                      onChange={(e) => setImageFile(e.target.files?.[0] || null)}
                    />
                    <button style={{ ...S.btn("outline", true), marginBottom: 10 }} onClick={() => imageRef.current?.click()}>
                      {imageFile ? `Image: ${imageFile.name}` : "Upload ECG Image"}
                    </button>
                  </>
                )}

                <div style={{ display: "flex", gap: 8 }}>
                  <button style={S.btn("primary")} disabled={loading} onClick={analyze}>{loading ? "Analyzing..." : "Run Analysis"}</button>
                  <button style={S.btn("outline")} onClick={reset}>Clear</button>
                </div>
              </div>
            </>
          ) : (
            <div style={S.card}>
              <div style={{ fontSize: 14, color: C.textMuted, marginBottom: 6 }}>Primary Diagnosis</div>
              <div style={{ fontSize: 28, fontWeight: 900, color: CLASSES.find((c) => c.key === result.result)?.color || C.text }}>{result.result}</div>
              <div style={{ marginTop: 12, marginBottom: 12 }}>
                {CLASSES.map((c) => (
                  <div key={c.key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    <span style={{ width: 40, color: c.color, fontWeight: 700 }}>{c.key}</span>
                    <div style={{ flex: 1, height: 6, background: C.border, borderRadius: 3 }}>
                      <div style={{ width: `${result.scores?.[c.key] || 0}%`, height: "100%", background: c.color, borderRadius: 3 }} />
                    </div>
                    <span style={{ width: 40, textAlign: "right", color: C.textMuted }}>{result.scores?.[c.key] || 0}%</span>
                  </div>
                ))}
              </div>
              <button style={S.btn()} onClick={reset}>New Analysis</button>
            </div>
          )}
        </div>

        <AIChatPanel pathology={result?.result} patientContext={[patientRef && `Patient: ${patientRef}`, age && `Age: ${age}`, sex && `Sex: ${sex}`].filter(Boolean).join(", ")} />
      </div>
    </>
  );
}
