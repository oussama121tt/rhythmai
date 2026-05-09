import React, { useState, useRef, useEffect } from "react";
import { askAI } from "../utils/api";
import { CLASSES, C } from "../utils/constants";

export default function AIChatPanel({ pathology, patientContext = "" }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello, I am RhythmAI Assistant. Ask me about this ECG result, risk interpretation, or next clinical checks.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef(null);

  const pc = CLASSES.find((c) => c.key === pathology);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const next = [...messages, { role: "user", content: text }];
    setMessages(next);
    setInput("");
    setLoading(true);

    const systemPrompt = [
      "You are a concise clinical ECG assistant for cardiology decision support.",
      "Never provide definitive diagnosis; recommend clinical validation.",
      pathology ? `Current predicted class: ${pathology}.` : "",
      patientContext ? `Patient context: ${patientContext}.` : "",
    ]
      .filter(Boolean)
      .join(" ");

    try {
      const reply = await askAI(next, systemPrompt, {
        pathology,
        patientContext,
      });
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${e.message || "Unable to reach AI service."}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        width: 360,
        minWidth: 300,
        maxWidth: 420,
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        boxShadow: C.shadow,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "12px 14px",
          borderBottom: `1px solid ${C.border}`,
          background: C.gradientSoft,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
        }}
      >
        <div>
          <div style={{ fontSize: 14, fontWeight: 800, color: C.text }}>AI Clinical Copilot</div>
          <div style={{ fontSize: 12, color: C.textMuted }}>
            {pc ? `Focus: ${pc.key} · ${pc.label}` : "General ECG discussion"}
          </div>
        </div>
      </div>

      <div ref={listRef} style={{ flex: 1, overflowY: "auto", padding: 12, maxHeight: 420 }}>
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              marginBottom: 10,
              display: "flex",
              justifyContent: m.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                maxWidth: "88%",
                fontSize: 13,
                lineHeight: 1.5,
                padding: "8px 10px",
                borderRadius: 10,
                border: `1px solid ${m.role === "user" ? `${C.accent}33` : C.border}`,
                background: m.role === "user" ? C.accentLight : C.surface,
                color: C.text,
                whiteSpace: "pre-wrap",
              }}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && <div style={{ fontSize: 12, color: C.textMuted }}>RhythmAI is typing...</div>}
      </div>

      <div style={{ borderTop: `1px solid ${C.border}`, padding: 10, display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") send();
          }}
          placeholder="Ask about this result..."
          style={{
            flex: 1,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "8px 10px",
            fontSize: 13,
            outline: "none",
          }}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          style={{
            border: "none",
            borderRadius: 8,
            padding: "8px 12px",
            fontSize: 13,
            fontWeight: 700,
            background: C.gradient,
            color: "#fff",
            cursor: "pointer",
            opacity: loading || !input.trim() ? 0.6 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
